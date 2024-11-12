import zipfile
import os
from tqdm import tqdm

# 用于存储压缩文件名的列表
panorama_files = []
perspective_full_files = []

# 读取文件并提取链接
with open('download/README.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        if line.startswith('https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_'):
            panorama_files.append(line.strip().split('/')[-1])
        elif line.startswith('https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_perspective_full_'):
            perspective_full_files.append(line.strip().split('/')[-1])

def unzip_file(file_path, output_dir):

    if file.endswith('.zip'):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            total_files = len(zip_ref.infolist())
            with tqdm(total=total_files, unit='file', desc=f"解压 {file}") as pbar:
                for member in zip_ref.infolist():
                    zip_ref.extract(member, output_dir)
                    pbar.update(1)
    else:
        print("Unhandled file type")

# 保存压缩包的目录
save_pth = 'D:/dataset/S3D'

# 解压 Panorama压缩包
for link in panorama_files:
    file_name = link.split('/')[-1]
    file_path = os.path.join(save_pth, file_name)
    unzip_file(file_path, save_pth)

# 解压 Perspective (full)压缩包
for link in perspective_full_files:
    file_name = link.split('/')[-1]
    file_path = os.path.join(save_pth, file_name)
    unzip_file(file_path, save_pth)
