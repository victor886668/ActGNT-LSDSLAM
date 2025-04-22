import os
import re
import shutil
import json
import numpy as np

# 设置原文件夹路径和新文件夹路径
source_folder = './images_src'  # 替换为原文件夹路径
destination_folder = './images_rename0'  # 替换为目标文件夹路径

# 创建目标文件夹（如果目标文件夹不存在）
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 获取源文件夹中的所有文件
files = os.listdir(source_folder)

# 筛选出所有的 .jpg 文件
jpg_files = [file for file in files if file.endswith('.jpg')]

# 使用正则表达式提取文件中的时间戳部分
# def extract_timestamp(filename):
#     match = re.search(r'n\d{3}-(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}\+\d{4})', filename)
#     if match:
#         return match.group(1)
#     return None

# 对文件按时间戳进行排序
jpg_files= np.sort(jpg_files)

# 准备一个字典来存储原文件名和新文件名的对应关系
rename_dict = {}

# 遍历排序后的文件列表，拷贝并重命名文件
for index, file in enumerate(jpg_files, start=1):
    new_name = f"{index:04d}.jpg"  # 重命名为 0001.jpg, 0002.jpg 等
    old_path = os.path.join(source_folder, file)
    new_path = os.path.join(destination_folder, new_name)
    
    # 拷贝文件到目标文件夹
    shutil.copy2(old_path, new_path)  # 使用 copy2 保留文件的元数据（如修改时间）
    
    # 记录对应关系
    rename_dict[file] = new_name

# 保存对应关系到 JSON 文件
#json_file = os.path.join(destination_folder, 'rename_mapping.json')
#with open(json_file, 'w') as f:
#    json.dump(rename_dict, f, indent=4)

print(f"文件已拷贝到 {destination_folder} 并保存了对应关系到 {json_file}")
