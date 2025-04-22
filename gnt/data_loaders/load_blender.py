import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1):
    # 定义数据集划分类型
    splits = ['train', 'holdout', 'val', 'test']
    # 存储每个划分的元数据
    metas = {}
    # 读取每个划分的transforms.json文件
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    # 存储所有图像和相机姿态
    all_imgs = []
    all_poses = []
    # 记录每个划分的累积图像数量
    counts = [0]
    # 遍历每个划分
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        # 根据不同划分设置跳帧数
        if s=='train' or testskip==0:
            skip = 1
        elif s=='holdout':
            skip = 1
        else:
            skip = testskip
            
        # 读取图像和相机姿态
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        # 将图像归一化到[0,1]范围,保持RGBA四通道
        imgs = (np.array(imgs) / 255.).astype(np.float32) 
        # 转换相机姿态为float32类型
        poses = np.array(poses).astype(np.float32)
        # 更新累积计数
        counts.append(counts[-1] + imgs.shape[0])
        # 添加到总列表
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    # 为每个划分创建索引数组
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(4)]
    
    # 合并所有图像和姿态
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    # 获取图像尺寸和相机参数
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    # 创建用于渲染的相机轨迹
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    # 如果需要降采样
    if half_res:
        # 将分辨率减半
        H = H//2
        W = W//2
        focal = focal/2.

        # 创建降采样后的图像数组
        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        
        # 对每张图像进行降采样
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (H, W), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        
    # 返回处理后的数据
    return imgs, poses, render_poses, [H, W, focal], i_split


