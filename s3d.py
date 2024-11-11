import platform


# 获取系统类型
system_type = platform.system()
# 本地路径和服务器路径

if system_type == 'Windows':
    s3d_data_pth = "e:/datasets/Structure3D/Structured3D"
    s3d_bev_pth = "e:/datasets/Structure3D_bev/Structured3D"
    s3d_map_pth = "e:/datasets/Structure3D_map/Structured3D"
    s3d_scene_annos_loss = "logs/scene_annos.txt"
    s3d_scene_line_err = "logs/scene_line_err.txt"
    s3d_scene_obs_err = "logs/scene_observation.txt"
    s3d_csv_pth = "e:/datasets/Structure3D_csv/Structured3D"
else:
    s3d_data_pth = "/data1/chenjiwei/S3D/zip/Structured3D"
    s3d_bev_pth = "/data1/chenjiwei/S3D/Structure3D_bev/Structured3D"
    s3d_map_pth = "/data1/chenjiwei/S3D/Structure3D_map/Structured3D"
    s3d_scene_annos_loss = "/home/chenjiwei/Workspace/Structured3D/logs/scene_annos.txt"
    s3d_scene_line_err = "/home/chenjiwei/Workspace/Structured3D/logs/scene_line_err.txt"
    s3d_scene_obs_err = "/home/chenjiwei/Workspace/Structured3D/logs/scene_observation.txt"
    s3d_csv_pth = "/data1/chenjiwei/S3D/Structure3D_csv/Structured3D"

resolution = 25

# 标注数据缺失的场景
with open(s3d_scene_annos_loss, encoding="utf-8") as f:
    scene_invalid = f.readlines()
for index, item in enumerate(scene_invalid):
    scene_invalid[index] = item.replace("\n", "")
# 边线错误的场景
with open(s3d_scene_line_err, encoding="utf-8") as f:
    scene_invalid_append = f.readlines()
for index, item in enumerate(scene_invalid_append):
    scene_invalid_append[index] = item.replace("\n", "")
scene_invalid = scene_invalid + scene_invalid_append

# 观测数据缺失的场景
with open(s3d_scene_obs_err, encoding="utf-8") as f:
    sample_invalid = f.readlines()
for index, item in enumerate(sample_invalid):
    sample_invalid[index] = item.replace("\n", "")
