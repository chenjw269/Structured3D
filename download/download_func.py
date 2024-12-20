# 20241017
# 遍历下载链接进行下载

import requests
import os
from tqdm import tqdm

# # 用于存储下载链接的列表
# panorama_links = []
# perspective_full_links = []

# # 读取文件并提取链接
# with open('download/README.txt', 'r') as file:
#     lines = file.readlines()
#     for line in lines:
#         if line.startswith('https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_'):
#             panorama_links.append(line.strip())
#         elif line.startswith('https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_perspective_full_'):
#             perspective_full_links.append(line.strip())

# 下载文件的函数
def download_file(url, file_path):
    print(f"Download {url} \nSave to {file_path}")
    while True:
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
            with open(file_path, 'wb') as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
            progress_bar.close()
            print(f"下载成功: {file_path}")
            break
        except requests.exceptions.RequestException as e:
            print(f"下载失败: {e}, 正在重试...")

# # 创建保存压缩包的目录
# save_pth = 'D:/dataset/S3D'
# if not os.path.exists(save_pth):
#     os.makedirs(save_pth)

# # 下载 Panorama压缩包
# for link in panorama_links:
#     file_name = link.split('/')[-1]
#     file_path = os.path.join(save_pth, file_name)
#     download_file(link, file_path)

# # 下载 Perspective (full)压缩包
# for link in perspective_full_links:
#     file_name = link.split('/')[-1]
#     file_path = os.path.join(save_pth, file_name)
#     download_file(link, file_path)
    
# 已经下载成功的
# panorama
# 00 06 08 11
# perspective
# 01 02 03 04 05
