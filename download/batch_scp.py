import subprocess
import os
import pexpect

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


def batch_scp_to_server(file_path, remote_pth):
    server_user = 'chenjiwei'
    server_ip = '172.25.73.13'
    password = 'Cjw@121'

    if file.endswith('.zip'):
        scp_command = f'scp {file_path} {server_user}@{server_ip}:{remote_pth}'
        child = pexpect.spawn(scp_command)
        try:
            child.expect('password:')
            child.sendline(password)
            child.expect(pexpect.EOF)
            print(f"成功传输 {file} 到服务器")
        except pexpect.TIMEOUT:
            print(f"传输 {file} 到服务器失败: 超时")
        except pexpect.EOF:
            print(f"传输 {file} 到服务器失败: 意外结束")

# 保存压缩包的目录
save_pth = 'D:/dataset/S3D'
remote_pth = '/data1/chenjiwei/S3D'


# 传输 Panorama压缩包
for link in panorama_files:
    file_name = link.split('/')[-1]
    file_path = os.path.join(save_pth, file_name)
    batch_scp_to_server(file_path, remote_pth)

# 传输 Perspective (full)压缩包
for link in perspective_full_files:
    file_name = link.split('/')[-1]
    file_path = os.path.join(save_pth, file_name)
    batch_scp_to_server(file_path, remote_pth)