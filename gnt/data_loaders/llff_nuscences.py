import os
import numpy as np
import imageio
import torch
import sys

sys.path.append("../")
from torch.utils.data import Dataset
from .data_utils import random_crop, get_nearest_pose_ids
from .llff_data_utils import load_llff_data, batch_parse_llff_poses


class LLFFNuscencesDataset(Dataset):
    def __init__(self, args, mode, scenes=(), random_crop=True, **kwargs):
        self.folder_path = os.path.join(args.rootdir, "data/nerf_llff_data/")
        self.args = args
        self.mode = mode  # train / test / validation
        self.num_source_views = args.num_source_views
        self.random_crop = random_crop
        self.render_rgb_files = []
        self.render_intrinsics = []
        self.render_poses = []
        self.render_train_set_ids = []
        self.render_depth_range = []

        self.train_intrinsics = []
        self.train_poses = []
        self.train_rgb_files = []
        self.near_depth = 0
        self.far_depth = 0
        self.intrinsics = []
        self.c2w_mats = []

        
        
        hold_num = 20

        all_scenes = os.listdir(self.folder_path)
        if len(scenes) > 0:
            if isinstance(scenes, str):
                scenes = [scenes]
        else:
            scenes = all_scenes

        print("loading {} for {}".format(scenes, mode))
        for i, scene in enumerate(scenes):
            scene_path = os.path.join(self.folder_path, scene)
            _, poses, bds, render_poses, i_test, rgb_files = load_llff_data(
                scene_path, load_imgs=False, factor=4
            )
            near_depth = np.min(bds)
            far_depth = np.max(bds)
            self.far_depth = far_depth
            self.near_depth = near_depth
            intrinsics, c2w_mats = batch_parse_llff_poses(poses)
            

            i_test = np.arange(poses.shape[0])[:: self.args.llffhold]
            
           
            i_train = np.array(
                [
                    j
                    for j in np.arange(int(poses.shape[0]))
                    if (j not in i_test and j not in i_test)
                ]
            )
            
            # only for val
            # i_test = np.arange(poses.shape[0])
            # i_train = np.arange(poses.shape[0])
            
            # 
            if mode =='train':
                i_train = i_train[:hold_num]
                i_render = i_train
                
            elif mode =='holdout':
                i_train = i_train[hold_num:]
                i_render = i_train
            else :
                i_render = i_test
            
            
                
            

            self.train_intrinsics.append(intrinsics[i_train])
            self.train_poses.append(c2w_mats[i_train])
            self.train_rgb_files.append(np.array(rgb_files)[i_train].tolist())
            num_render = len(i_render)
            self.render_rgb_files.extend(np.array(rgb_files)[i_render].tolist())
            self.intrinsics = intrinsics
            self.render_intrinsics.extend([intrinsics_ for intrinsics_ in intrinsics[i_render]])
            self.render_poses.extend([c2w_mat for c2w_mat in c2w_mats[i_render]])
            self.render_depth_range.extend([[near_depth, far_depth]] * num_render)
            self.render_train_set_ids.extend([i] * num_render)


            # tmp1 = self.train_intrinsics[0]
            # tmp2 = np.array(self.render_intrinsics)
            # tmp = tmp1-tmp2
            # print(np.array_equal(tmp1, tmp2))  # True
            #break
            
            print("loading {} file".format(num_render))


    def __len__(self):
        return (
            #len(self.render_rgb_files) * 100000
            len(self.render_rgb_files)
            if self.mode == "train"
            else len(self.render_rgb_files)
        )
    
    def add_mixdata(self, render_rgb_files, render_poses, render_intrinsics):
        
        print("============= 数据混合前： ==============",len(self.render_poses))
        self.render_rgb_files.extend(render_rgb_files)
        self.render_poses.extend(render_poses)
        self.render_intrinsics.extend(render_intrinsics)
        print("============= 数据混合后： ==============",len(self.render_poses))
        
        #update
        num_render = len(self.render_rgb_files)
        new_render_depth_range = []
        new_render_train_set_ids = []
        new_render_depth_range.extend([[self.near_depth, self.far_depth]] * num_render)
        new_render_train_set_ids.extend([0] * num_render)
        self.render_depth_range = new_render_depth_range
        self.render_train_set_ids = new_render_train_set_ids
        
        train_intrinsics = []
        train_poses = []
       
        train_intrinsics.append(np.array(self.render_intrinsics))
        train_poses.append(np.array(self.render_poses))
        train_rgb_files = []
        train_rgb_files.append(self.render_rgb_files)
        self.train_intrinsics = train_intrinsics
        self.train_poses = train_poses
        self.train_rgb_files = train_rgb_files
        
        print("============= 数据混合完成 ==============")
    
    def delete_indexs(self, used_id):
        new_render_rgb_files = []
        new_render_poses = []
        new_render_intrinsics = []

        render_rgb_files = []
        render_poses = []
        render_intrinsics = []
        depth_range = []
        print("============= 数据删除前： ==============",len(self.render_poses))
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
        
        #update
        num_render = len(self.render_rgb_files)
        new_render_depth_range = []
        new_render_train_set_ids = []
        new_render_depth_range.extend([[self.near_depth, self.far_depth]] * num_render)
        new_render_train_set_ids.extend([0]*num_render)
        self.render_depth_range = new_render_depth_range
        self.render_train_set_ids = new_render_train_set_ids
        
        train_intrinsics = []
        train_poses = []

        train_intrinsics.append(np.array(self.render_intrinsics))
        train_poses.append(np.array(self.render_poses))

        train_rgb_files = []
        train_rgb_files.append(self.render_rgb_files)
        self.train_intrinsics = train_intrinsics
        self.train_poses = train_poses
        self.train_rgb_files = train_rgb_files
        
        
        print("============= 数据删除后： ==============",len(self.render_poses))
        print("============= 数据删除完成 ==============")
        return render_rgb_files, render_poses, render_intrinsics    
    
    

    def __getitem__(self, idx):
        idx = idx % len(self.render_rgb_files)
        rgb_file = self.render_rgb_files[idx]
        print('rgb_file:',rgb_file)
        rgb = imageio.imread(rgb_file).astype(np.float32) / 255.0
        
        render_pose = self.render_poses[idx]
        intrinsics = self.render_intrinsics[idx]
        #print('idx:',idx)
        depth_range = self.render_depth_range[idx]

        train_set_id = self.render_train_set_ids[idx]
        
        train_rgb_files = self.train_rgb_files[train_set_id]
        # print('train_rgb_files:',len(train_rgb_files))
        # print('train_rgb_files:',train_rgb_files[0].tpye)
        # print('train_rgb_files:',train_rgb_files[0])
        train_poses = self.train_poses[train_set_id]
        train_intrinsics = self.train_intrinsics[train_set_id]

        img_size = rgb.shape[:2]
        camera = np.concatenate(
            (list(img_size), intrinsics.flatten(), render_pose.flatten())
        ).astype(np.float32)

        if self.mode == "train":
            if rgb_file in train_rgb_files:
                id_render = train_rgb_files.index(rgb_file)
            else:
                id_render = -1
            subsample_factor = np.random.choice(np.arange(1, 4), p=[0.2, 0.45, 0.35])
            num_select = self.num_source_views + np.random.randint(low=-2, high=2)
        else:
            id_render = -1
            subsample_factor = 1
            num_select = self.num_source_views

        nearest_pose_ids = get_nearest_pose_ids(
            render_pose,
            train_poses,
            min(self.num_source_views * subsample_factor, 28),
            tar_id=id_render,
            angular_dist_method="dist",
        )
        nearest_pose_ids = np.random.choice(
            nearest_pose_ids, min(num_select, len(nearest_pose_ids)), replace=False
        )

        assert id_render not in nearest_pose_ids
        # occasionally include input image
        if np.random.choice([0, 1], p=[0.995, 0.005]) and self.mode == "train":
            nearest_pose_ids[np.random.choice(len(nearest_pose_ids))] = id_render

        src_rgbs = []
        src_cameras = []
        for id in nearest_pose_ids:
            # print('nearest_pose_ids:',nearest_pose_ids)
            # print('cur_id:',id)
            src_rgb = imageio.imread(train_rgb_files[id]).astype(np.float32) / 255.0
            train_pose = train_poses[id]
            train_intrinsics_ = train_intrinsics[id]

            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate(
                (list(img_size), train_intrinsics_.flatten(), train_pose.flatten())
            ).astype(np.float32)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)
        if self.mode == "train" and self.random_crop:
            crop_h = np.random.randint(low=250, high=750)
            crop_h = crop_h + 1 if crop_h % 2 == 1 else crop_h
            crop_w = int(400 * 600 / crop_h)
            crop_w = crop_w + 1 if crop_w % 2 == 1 else crop_w
            rgb, camera, src_rgbs, src_cameras = random_crop(
                rgb, camera, src_rgbs, src_cameras, (crop_h, crop_w)
            )

        depth_range = torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.6])
        return {
            "ids": idx,
            "rgb": torch.from_numpy(rgb[..., :3]),
            "camera": torch.from_numpy(camera),
            "rgb_path": rgb_file,
            "src_rgbs": torch.from_numpy(src_rgbs[..., :3]),
            "src_cameras": torch.from_numpy(src_cameras),
            "depth_range": depth_range,
        }