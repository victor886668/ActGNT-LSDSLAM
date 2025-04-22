import os
import time
import numpy as np
import shutil
import torch
import torch.utils.data.distributed
import cv2



if __name__ == "__main__":
    
    path = '/root/autodl-tmp/afgt/data/nerf_llff_data/nuscenes/images'
    file_list = os.listdir(path)
    
    for id in range(len(file_list)):
        #print(file_list[id])
        img_path =os.path.join(path,file_list[id])
        img = cv2.imread(img_path,-1)
        # print(file_list[id])
        # print('id:',id)
        print('img:',img.shape)
        #break
