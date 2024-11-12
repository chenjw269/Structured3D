# 整理下载链接

# 用于存储下载链接的列表
panorama_links = []
perspective_full_links = []

# 读取文件并提取链接
with open('download/README.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        if line.startswith('https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_'):
            panorama_links.append(line)
        elif line.startswith('https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_perspective_full_'):
            perspective_full_links.append(line)

# panorama 下载链接
with open("download/pano_url.txt", "w") as f:
    f.writelines(panorama_links)

# perspective full 下载链接
with open("download/pesp_full_url.txt", "w") as f:
    f.writelines(perspective_full_links)
