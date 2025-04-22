import os
import re
import imageio
import cv2
import numpy as np

def convert_pfm_to_tiff(src_dir, dst_dir):
    """
    将源目录下的所有PFM文件转存为EXR格式到目标目录
    :param src_dir: 源目录路径（包含PFM文件）
    :param dst_dir: 目标目录路径（输出EXR文件）
    """
    # 确保目标目录存在
    os.makedirs(dst_dir, exist_ok=True)
    
    # 匹配文件名模式（如val_0001_fine_uncert_src.pfm）
    pattern = re.compile(r'val_(\d+)_fine_uncert_src\.pfm$')
    
    # 遍历源目录
    for filename in sorted(os.listdir(src_dir)):
        match = pattern.match(filename)
        if not match:
            continue
            
        # 构造完整路径
        pfm_path = os.path.join(src_dir, filename)
        exr_filename = f"val_{match.group(1)}_fine_uncert_src.tiff"
        exr_path = os.path.join(dst_dir, exr_filename)
        
        try:
            # 读取PFM文件（自动转换为numpy数组）
            pfm_data = imageio.imread(pfm_path)
            
            # 转换为OpenCV兼容格式（保持float32）
            if pfm_data.ndim == 2:  # 单通道
                cv_data = pfm_data.astype(np.float32)
            else:  # 多通道（假设需要RGB->BGR转换）
                cv_data = cv2.cvtColor(pfm_data, cv2.COLOR_RGB2BGR)
            
            # 保存为EXR文件
            cv2.imwrite(exr_path, cv_data)
            print(f"转换成功: {filename} -> {exr_filename}")
            
        except Exception as e:
            print(f"转换失败 {filename}: {str(e)}")

if __name__ == "__main__":
    # 配置路径（根据实际情况修改）
    source_directory = "/home/slam/datasets/test/eval"    # PFM文件所在目录
    target_directory = "/home/slam/datasets/test/eval_exr"   # 输出EXR目录
    
    convert_pfm_to_tiff(source_directory, target_directory)