import sys
sys.path.append(".")

import os
from download.download_func import download_file


with open("download/pano_url.txt") as f:
    panorama_links = f.readlines()

# 创建保存压缩包的目录
save_pth = 'D:/dataset/S3D'
if not os.path.exists(save_pth):
    os.makedirs(save_pth)

# 下载 Panorama压缩包
for link in panorama_links[0:6]:
    link = link.strip()
    file_name = link.split('/')[-1]
    file_path = os.path.join(save_pth, file_name)
    download_file(link, file_path)
