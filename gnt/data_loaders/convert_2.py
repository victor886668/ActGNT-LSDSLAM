import os
import json
import numpy as np
import shutil
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from PIL import Image
import random

def convert_nuscenes_to_nerf(dataroot, version="v1.0-mini", camera_name="CAM_FRONT", output_dir='./nerf_data', num_samples=404):
    """
    Convert NuScenes CAM_FRONT data into NeRF Synthetic format with a custom dataset split.
    Training: 90% (20% 非 holdout, 70% holdout)
    Validation: 10%
    """
    if not os.path.exists(dataroot):
        raise FileNotFoundError(f"❌ NuScenes 数据集路径不存在: {dataroot}")

    print(f"📂 Loading NuScenes dataset from: {dataroot}")
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    frames = []

    # 处理数据
    for sample_idx in range(min(num_samples, len(nusc.sample))):
        sample = nusc.sample[sample_idx]
        camera_data = nusc.get('sample_data', sample['data'][camera_name])
        image_path = os.path.join(dataroot, camera_data['filename'])

        # 转换 JPG 为 PNG 并保存
        img = Image.open(image_path)
        output_image_path = os.path.join(images_dir, f'r_{sample_idx}.png')
        img.save(output_image_path)

        # 读取相机内参
        calib_data = nusc.get('calibrated_sensor', camera_data['calibrated_sensor_token'])
        intrinsics = np.array(calib_data['camera_intrinsic'])
        focal_length = intrinsics[0, 0]
        image_width = camera_data['width']
        camera_angle_x = 2 * np.arctan(image_width / (2 * focal_length))

        # 读取相机位姿
        ego_pose = nusc.get('ego_pose', camera_data['ego_pose_token'])
        ego_translation = np.array(ego_pose['translation'])
        ego_rotation = Quaternion(ego_pose['rotation']).rotation_matrix

        calib_translation = np.array(calib_data['translation'])
        calib_rotation = Quaternion(calib_data['rotation']).rotation_matrix

        # 计算相机到世界坐标的变换矩阵
        global_rotation = ego_rotation @ calib_rotation
        global_translation = ego_rotation @ calib_translation + ego_translation

        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = global_rotation.T
        transform_matrix[:3, 3] = -global_rotation.T @ global_translation

        # 交换 Y-Z 轴，使其适配 NeRF
        swap_yz = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        transform_matrix = swap_yz @ transform_matrix

        # 记录到 frames 列表
        frames.append({
            'file_path': f'./images/r_{sample_idx}',
            'rotation': 0,
            'transform_matrix': transform_matrix.astype(np.float32).tolist()
        })

    # 数据集划分：
    # 训练集 (90%) → 20% 非 holdout, 70% holdout
    # 验证集 (10%)

    random.shuffle(frames)
    num_train = int(0.9 * len(frames))
    num_val = int(0.1 * len(frames))

    train_frames = frames[:num_train]  # 90% 训练数据
    val_frames = frames[num_train:num_train + num_val]  # 10% 验证数据

    # 进一步划分 训练集 → 20% 非 holdout, 70% holdout
    num_train_non_holdout = int(0.2 * len(train_frames))  # 20% 直接训练
    train_non_holdout = train_frames[:num_train_non_holdout]  # 固定训练数据
    train_holdout = train_frames[num_train_non_holdout:]  # 需要 Active Learning 选取的训练数据

    # 训练全集 = 非 holdout + holdout
    train_all = train_non_holdout + train_holdout

    # 保存 JSON 文件
    def save_json(filename, frames_data):
        with open(os.path.join(output_dir, filename), "w") as f:
            json.dump({
                'camera_angle_x': camera_angle_x,
                'frames': frames_data
            }, f, indent=4)
        print(f"✅ Saved {filename}")

    save_json('transforms_train.json', train_non_holdout)  # 训练集（20% 非 holdout）
    save_json('transforms_holdout.json', train_holdout)  # 训练集（70% holdout）
    save_json('transforms_trains.json', train_all)  # 训练全集（90% = 20% 非 holdout + 70% holdout）
    save_json('transforms_val.json', val_frames)  # 验证集（10%）

    print(f"✅ Conversion complete! Data saved to {output_dir}")

if __name__ == "__main__":
    # 你可以在这里修改 `dataroot` 以匹配你的 NuScenes 数据集路径
    convert_nuscenes_to_nerf(
        dataroot = r'/root/autodl-tmp/GNT-code/GNT-code/v1.0-mini',  # ⚠️ 修改为NuScenes 数据路径
        version="v1.0-mini",
        camera_name="CAM_FRONT",  # 可以改成 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT' 等
        output_dir="/root/autodl-tmp/GNT-code/GNT-code/nerf_synthetic/nerf_data",  # NeRF 数据存放位置
        num_samples=404  # mini 版最多 404 张图
    )
