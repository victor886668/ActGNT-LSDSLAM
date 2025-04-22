import os
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
import sys
import json

sys.path.append("../")
from .data_utils import rectify_inplane_rotation, get_nearest_pose_ids


def read_cameras(pose_file):
    basedir = os.path.dirname(pose_file)
    with open(pose_file, "r") as fp:
        meta = json.load(fp)

    camera_angle_x = float(meta["camera_angle_x"])
    rgb_files = []
    c2w_mats = []

    img = imageio.imread(os.path.join(basedir, meta["frames"][0]["file_path"] + ".png"))
    H, W = img.shape[:2]
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
    intrinsics = get_intrinsics_from_hwf(H, W, focal)

    for i, frame in enumerate(meta["frames"]):
        rgb_file = os.path.join(basedir, meta["frames"][i]["file_path"][2:] + ".png")
        rgb_files.append(rgb_file)
        c2w = np.array(frame["transform_matrix"])
        w2c_blender = np.linalg.inv(c2w)
        w2c_opencv = w2c_blender
        w2c_opencv[1:3] *= -1
        c2w_opencv = np.linalg.inv(w2c_opencv)
        c2w_mats.append(c2w_opencv)
    c2w_mats = np.array(c2w_mats)
    return rgb_files, np.array([intrinsics] * len(meta["frames"])), c2w_mats, focal


def get_intrinsics_from_hwf(h, w, focal):
    return np.array(
        [[focal, 0, 1.0 * w / 2, 0], [0, focal, 1.0 * h / 2, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )


class NerfNuscenceDataset(Dataset):
    def __init__(
        self,
        args,
        mode,
        # scenes=('chair', 'drum', 'lego', 'hotdog', 'materials', 'mic', 'ship'),
        scenes=(),  # 场景列表,默认为空元组
        **kwargs
    ):
        # 设置数据集根目录路径
        self.folder_path = os.path.join(args.rootdir, "mini_nuscene")
        # 是否需要校正平面内旋转
        self.rectify_inplane_rotation = args.rectify_inplane_rotation
        # 将validation模式转换为val模式
        if mode == "validation":
            mode = "val"
        # 确保模式是train/val/test之一
        assert mode in ["train", "val"]
        self.mode = mode  # 设置数据集模式
        # 设置源视图数量
        self.num_source_views = args.num_source_views
        # 设置测试时的跳帧数
        self.testskip = args.testskip

        # 定义所有可用场景
        # all_scenes = ("chair", "drums", "lego", "hotdog", "materials", "mic", "ship")
        # 如果指定了场景,使用指定场景,否则使用所有场景
        # if len(scenes) > 0:
        #     if isinstance(scenes, str):
        #         scenes = [scenes]
        # else:
        #     scenes = all_scenes
        pose_file = os.path.join(self.folder_path, "transforms_{}.json".format(self.mode))
        print("loading {}".format(pose_file))
        # 初始化渲染数据列表
        self.render_rgb_files = []  # RGB图像文件路径
        self.render_poses = []      # 相机姿态
        self.render_intrinsics = [] # 相机内参

        # 根据模式读取对应的transforms文件
        rgb_files, intrinsics, poses, focal = read_cameras(pose_file)
        # 非训练模式时进行跳帧采样
        if self.mode != "train":
            rgb_files = rgb_files[:: self.testskip]
            intrinsics = intrinsics[:: self.testskip]
            poses = poses[:: self.testskip]
        # 将数据添加到列表中
        self.render_rgb_files.extend(rgb_files)
        self.render_poses.extend(poses)
        self.render_intrinsics.extend(intrinsics)

    def __len__(self):
        # 返回数据集大小
        return len(self.render_rgb_files)
    def add_mixdata(self, render_rgb_files, render_poses, render_intrinsics):
        self.render_rgb_files.extend(render_rgb_files)
        self.render_poses.extend(render_poses)
        self.render_intrinsics.extend(render_intrinsics)
        print("============= 数据混合完成 ==============")


    def __getitem__(self, idx):
        # 获取当前索引的渲染数据
        rgb_file = self.render_rgb_files[idx]
        render_pose = self.render_poses[idx]
        render_intrinsics = self.render_intrinsics[idx]

        # 获取训练数据文件路径
        train_pose_file = os.path.join("/".join(rgb_file.split("/")[:-2]), "transforms_trains.json")
        train_rgb_files, train_intrinsics, train_poses, focal = read_cameras(train_pose_file)

        # 训练模式下获取渲染ID和采样因子
        if self.mode == "train":
            # id_render = int(os.path.basename(rgb_file)[:-4].split("_")[1])
            id_render = idx
            # print("#"*78)
            # print(id_render)
            subsample_factor = np.random.choice(np.arange(1, 4), p=[0.3, 0.5, 0.2])
        else:
            id_render = -1
            subsample_factor = 1

      

        # 读取并处理RGB图像
        rgb = imageio.imread(rgb_file).astype(np.float32) / 255.0
        rgb = rgb[..., [-1]] * rgb[..., :3] + 1 - rgb[..., [-1]]  # alpha混合
        img_size = rgb.shape[:2]
        # 组合相机参数
        camera = np.concatenate(
            (list(img_size), render_intrinsics.flatten(), render_pose.flatten())
        ).astype(np.float32)

        # 获取最近的源视图ID
        nearest_pose_ids = get_nearest_pose_ids(
            render_pose,
            train_poses,
            int(self.num_source_views * subsample_factor),
            tar_id=id_render,
            angular_dist_method="vector",
        )
        # 随机选择源视图
        nearest_pose_ids = np.random.choice(nearest_pose_ids, self.num_source_views, replace=False)

        # 确保目标视图不在源视图中
        assert id_render not in nearest_pose_ids
        # 小概率将输入图像包含在源视图中
        if np.random.choice([0, 1], p=[0.995, 0.005]) and self.mode == "train":
            nearest_pose_ids[np.random.choice(len(nearest_pose_ids))] = id_render

        # 处理源视图数据
        src_rgbs = []
        src_cameras = []
        for id in nearest_pose_ids:
            # 读取并处理源视图RGB图像
            src_rgb = imageio.imread(train_rgb_files[id]).astype(np.float32) / 255.0
            src_rgb = src_rgb[..., [-1]] * src_rgb[..., :3] + 1 - src_rgb[..., [-1]]
            train_pose = train_poses[id]
            train_intrinsics_ = train_intrinsics[id]
            # 如果需要,校正平面内旋转
            if self.rectify_inplane_rotation:
                train_pose, src_rgb = rectify_inplane_rotation(train_pose, render_pose, src_rgb)

            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            # 组合源视图相机参数
            src_camera = np.concatenate(
                (list(img_size), train_intrinsics_.flatten(), train_pose.flatten())
            ).astype(np.float32)
            src_cameras.append(src_camera)

        # 堆叠源视图数据
        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)

        # 设置深度范围
        near_depth = 2.0
        far_depth = 6.0
        depth_range = torch.tensor([near_depth, far_depth])

        # 返回数据字典
        return {
            "rgb": torch.from_numpy(rgb[..., :3]),           # 目标RGB图像
            "camera": torch.from_numpy(camera),              # 目标相机参数
            "rgb_path": rgb_file,                           # RGB文件路径
            "src_rgbs": torch.from_numpy(src_rgbs[..., :3]), # 源视图RGB图像
            "src_cameras": torch.from_numpy(src_cameras),     # 源视图相机参数
            "depth_range": depth_range,                      # 深度范围
        }

class NerfNuscenceDatasetWithHoldout(Dataset):
    def __init__(
        self,
        args,
        mode='holdout',
        **kwargs
    ):
        # 设置数据集根目录路径
        # self.folder_path = os.path.join(args.rootdir, "data/nerf_synthetic_active/")
        self.folder_path = os.path.join(args.rootdir, "mini_nuscene")
        # 是否需要校正平面内旋转
        self.rectify_inplane_rotation = args.rectify_inplane_rotation
        # 将validation模式转换为val模式
        # if mode == "validation":
        #     mode = "val"
        # 确保模式是train/val/test之一
        assert mode in ["holdout"]
        self.mode = mode  # 设置数据集模式
        # 设置源视图数量
        self.num_source_views = args.num_source_views
        # 设置测试时的跳帧数
        self.testskip = args.testskip

        # 定义所有可用场景
        # all_scenes = ("chair", "drums", "lego", "hotdog", "materials", "mic", "ship")
        # 如果指定了场景,使用指定场景,否则使用所有场景
        # if len(scenes) > 0:
        #     if isinstance(scenes, str):
        #         scenes = [scenes]
        # else:
        #     scenes = all_scenes
        pose_file = os.path.join(self.folder_path, "transforms_{}.json".format(self.mode))
        print("loading {}".format(pose_file))
        # 初始化渲染数据列表
        self.render_rgb_files = []  # RGB图像文件路径
        self.render_poses = []      # 相机姿态
        self.render_intrinsics = [] # 相机内参

        # 根据模式读取对应的transforms文件
        rgb_files, intrinsics, poses, focal = read_cameras(pose_file)
        
        # 将数据添加到列表中
        self.render_rgb_files.extend(rgb_files)
        self.render_poses.extend(poses)
        self.render_intrinsics.extend(intrinsics)

    def __len__(self):
        # 返回数据集大小
        return len(self.render_rgb_files)
    
    def delete_indexs(self, used_id):
        new_render_rgb_files = []
        new_render_poses = []
        new_render_intrinsics = []

        render_rgb_files = []
        render_poses = []
        render_intrinsics = []
        
        for i in range(0, len(self.render_poses)):
            if i in used_id:
                render_rgb_files.append(self.render_rgb_files[i])
                render_poses.append(self.render_poses[i])
                render_intrinsics.append(self.render_intrinsics[i])
            else:
                new_render_rgb_files.append(self.render_rgb_files[i])
                new_render_intrinsics.append(self.render_intrinsics[i])
                new_render_poses.append(self.render_poses[i])

        # 重新定义序列
        self.render_poses = new_render_poses
        self.render_intrinsics = new_render_intrinsics
        self.render_rgb_files = new_render_rgb_files
        return render_rgb_files, render_poses, render_intrinsics
    def __getitem__(self, idx):
        # 获取当前索引的渲染数据
        rgb_file = self.render_rgb_files[idx]
        render_pose = self.render_poses[idx]
        render_intrinsics = self.render_intrinsics[idx]

        # 获取训练数据文件路径
        train_pose_file = os.path.join("/".join(rgb_file.split("/")[:-2]), f"transforms_{self.mode}.json")
        train_rgb_files, train_intrinsics, train_poses, focal = read_cameras(train_pose_file)

        # 训练模式下获取渲染ID和采样因子
        if self.mode == "train":
            id_render = int(os.path.basename(rgb_file)[:-4].split("_")[1])
            subsample_factor = np.random.choice(np.arange(1, 4), p=[0.3, 0.5, 0.2])
        else:
            id_render = -1
            subsample_factor = 1

        # 读取并处理RGB图像
        rgb = imageio.imread(rgb_file).astype(np.float32) / 255.0
        rgb = rgb[..., [-1]] * rgb[..., :3] + 1 - rgb[..., [-1]]  # alpha混合
        img_size = rgb.shape[:2]
        # 组合相机参数
        camera = np.concatenate(
            (list(img_size), render_intrinsics.flatten(), render_pose.flatten())
        ).astype(np.float32)

        # 获取最近的源视图ID
        nearest_pose_ids = get_nearest_pose_ids(
            render_pose,
            train_poses,
            int(self.num_source_views * subsample_factor),
            tar_id=id_render,
            angular_dist_method="vector",
        )
        # 随机选择源视图
        nearest_pose_ids = np.random.choice(nearest_pose_ids, self.num_source_views, replace=False)

        # 确保目标视图不在源视图中
        assert id_render not in nearest_pose_ids
        # 小概率将输入图像包含在源视图中
        if np.random.choice([0, 1], p=[0.995, 0.005]) and self.mode == "train":
            nearest_pose_ids[np.random.choice(len(nearest_pose_ids))] = id_render

        # 处理源视图数据
        src_rgbs = []
        src_cameras = []
        for id in nearest_pose_ids:
            # 读取并处理源视图RGB图像
            src_rgb = imageio.imread(train_rgb_files[id]).astype(np.float32) / 255.0
            src_rgb = src_rgb[..., [-1]] * src_rgb[..., :3] + 1 - src_rgb[..., [-1]]
            train_pose = train_poses[id]
            train_intrinsics_ = train_intrinsics[id]
            # 如果需要,校正平面内旋转
            if self.rectify_inplane_rotation:
                train_pose, src_rgb = rectify_inplane_rotation(train_pose, render_pose, src_rgb)

            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            # 组合源视图相机参数
            src_camera = np.concatenate(
                (list(img_size), train_intrinsics_.flatten(), train_pose.flatten())
            ).astype(np.float32)
            src_cameras.append(src_camera)

        # 堆叠源视图数据
        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)

        # 设置深度范围
        near_depth = 2.0
        far_depth = 6.0
        depth_range = torch.tensor([near_depth, far_depth])

        # 返回数据字典
        return {
            "ids": idx,
            "rgb": torch.from_numpy(rgb[..., :3]),           # 目标RGB图像
            "camera": torch.from_numpy(camera),              # 目标相机参数
            "rgb_path": rgb_file,                           # RGB文件路径
            "src_rgbs": torch.from_numpy(src_rgbs[..., :3]), # 源视图RGB图像
            "src_cameras": torch.from_numpy(src_cameras),     # 源视图相机参数
            "depth_range": depth_range,                      # 深度范围
        }


# class NerfSyntheticDatasetWithHoldout(Dataset):
#     def __init__(
#         self,
#         args,
#         mode,
#         scenes=(),
#         **kwargs
#     ):
#         self.folder_path = os.path.join(args.rootdir, "data/nerf_synthetic/")
#         self.rectify_inplane_rotation = args.rectify_inplane_rotation
#         if mode == "validation":
#             mode = "val"
#         assert mode in ["train", "val", "test", "holdout"]  # 添加holdout模式
#         self.mode = mode
#         self.num_source_views = args.num_source_views
#         self.testskip = args.testskip

#         all_scenes = ("chair", "drums", "lego", "hotdog", "materials", "mic", "ship")
#         if len(scenes) > 0:
#             if isinstance(scenes, str):
#                 scenes = [scenes]
#         else:
#             scenes = all_scenes

#         print("loading {} for {}".format(scenes, mode))
#         self.render_rgb_files = []
#         self.render_poses = []
#         self.render_intrinsics = []

#         for scene in scenes:
#             self.scene_path = os.path.join(self.folder_path, scene)
#             pose_file = os.path.join(self.scene_path, "transforms_{}.json".format(mode))
#             rgb_files, intrinsics, poses, focal = read_cameras(pose_file)
            
#             # 对不同模式使用不同的跳帧策略
#             if self.mode == "train" or self.testskip == 0:
#                 skip = 1
#             elif self.mode == "holdout":
#                 skip = 1
#             else:
#                 skip = self.testskip
                
#             rgb_files = rgb_files[::skip]
#             intrinsics = intrinsics[::skip]
#             poses = poses[::skip]
            
#             self.render_rgb_files.extend(rgb_files)
#             self.render_poses.extend(poses)
#             self.render_intrinsics.extend(intrinsics)
#         # 获取图像
#         self.images = [imageio.imread(rgb_file) for rgb_file in self.render_rgb_files]

#     def __len__(self):
#         return len(self.render_rgb_files)

#     def __getitem__(self, idx):
#         rgb_file = self.render_rgb_files[idx]
#         render_pose = self.render_poses[idx]
#         render_intrinsics = self.render_intrinsics[idx]

#         # 修改训练数据的获取逻辑，考虑holdout数据
#         train_pose_file = os.path.join("/".join(rgb_file.split("/")[:-2]), "transforms_train.json")
#         train_rgb_files, train_intrinsics, train_poses, focal = read_cameras(train_pose_file)

#         if self.mode == "train":
#             id_render = int(os.path.basename(rgb_file)[:-4].split("_")[1])
#             subsample_factor = np.random.choice(np.arange(1, 4), p=[0.3, 0.5, 0.2])
#         else:
#             id_render = -1
#             subsample_factor = 1

#         # 读取和处理图像，保持RGBA格式
#         rgb = imageio.imread(rgb_file).astype(np.float32) / 255.0
#         # 处理alpha通道
#         rgb = rgb[..., [-1]] * rgb[..., :3] + 1 - rgb[..., [-1]]
        
#         img_size = rgb.shape[:2]
#         camera = np.concatenate(
#             (list(img_size), render_intrinsics.flatten(), render_pose.flatten())
#         ).astype(np.float32)

#         nearest_pose_ids = get_nearest_pose_ids(
#             render_pose,
#             train_poses,
#             int(self.num_source_views * subsample_factor),
#             tar_id=id_render,
#             angular_dist_method="vector",
#         )
#         nearest_pose_ids = np.random.choice(nearest_pose_ids, self.num_source_views, replace=False)

#         assert id_render not in nearest_pose_ids
#         # 偶尔包含输入图像
#         if np.random.choice([0, 1], p=[0.995, 0.005]) and self.mode == "train":
#             nearest_pose_ids[np.random.choice(len(nearest_pose_ids))] = id_render

#         src_rgbs = []
#         src_cameras = []
#         for id in nearest_pose_ids:
#             src_rgb = imageio.imread(train_rgb_files[id]).astype(np.float32) / 255.0
#             src_rgb = src_rgb[..., [-1]] * src_rgb[..., :3] + 1 - src_rgb[..., [-1]]
#             train_pose = train_poses[id]
#             train_intrinsics_ = train_intrinsics[id]
            
#             if self.rectify_inplane_rotation:
#                 train_pose, src_rgb = rectify_inplane_rotation(train_pose, render_pose, src_rgb)

#             src_rgbs.append(src_rgb)
#             img_size = src_rgb.shape[:2]
#             src_camera = np.concatenate(
#                 (list(img_size), train_intrinsics_.flatten(), train_pose.flatten())
#             ).astype(np.float32)
#             src_cameras.append(src_camera)

#         src_rgbs = np.stack(src_rgbs, axis=0)
#         src_cameras = np.stack(src_cameras, axis=0)

#         near_depth = 2.0
#         far_depth = 6.0

#         depth_range = torch.tensor([near_depth, far_depth])

#         # 获取图像尺寸和相机参数
#         H, W = rgb.shape[:2]

#         return {
#             "rgb": torch.from_numpy(rgb[..., :3]),
#             "camera": torch.from_numpy(camera),
#             "rgb_path": rgb_file,
#             "src_rgbs": torch.from_numpy(src_rgbs[..., :3]),
#             "src_cameras": torch.from_numpy(src_cameras),
#             "depth_range": depth_range,
#             "focal": focal,
#             "H": H,
#             "W": W,
#         }
