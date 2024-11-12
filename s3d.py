import os.path
import platform


# 获取系统类型
system_type = platform.system()
# 本地路径和服务器路径

if system_type == 'Windows':
    s3d_download_url = ""

    # s3d_data_pth = "e:/datasets/Structure3D/Structured3D"
    s3d_pesp_data_pth = "d:/dataset/S3D/Structured3D"
    # s3d_bev_pth = "e:/datasets/Structure3D_bev/Structured3D"
    s3d_bev_pth = "d:/dataset/S3D/Structured3D_bev/"

    # cad 地图
    s3d_map_pth = "e:/datasets/Structure3D_map/Structured3D"
    s3d_annos_pth = "e:/datasets/Structure3D/Structured3D"

    # 场景标注缺失
    s3d_scene_annos_loss = "logs/scene_annos.txt"
    s3d_scene_line_err = "logs/scene_line_err.txt"

    # 观测数据缺失
    s3d_pano_obs_err = "logs/scene_observation.txt" # 全景图
    s3d_pesp_obs_err = "logs/pesp_obs_loss.txt" # 单视角视图

    # 数据集目录
    s3d_csv_pth = "e:/datasets/Structure3D_csv/Structured3D"
else:
    s3d_download_url = "/home/chenjiwei/Workspace/Structured3D/download/README.txt"
    s3d_download_pth = "/data1/chenjiwei/S3D"

    s3d_pano_data_pth = "/data1/chenjiwei/S3D/pano/Structured3D"
    s3d_pesp_data_pth = "/data1/chenjiwei/S3D/pesp/Structured3D"
    s3d_bev_pth = "/data1/chenjiwei/S3D/Structure3D_bev/Structured3D"

    # cad 地图
    s3d_map_pth = "/data1/chenjiwei/S3D/Structure3D_map/Structured3D"
    s3d_annos_pth = "e:/datasets/Structure3D/Structured3D"

    # 场景标注缺失
    s3d_project_pth = "/home/chenjiwei/Workspace/Structured3D"
    s3d_scene_annos_loss = os.path.join(
        s3d_project_pth, "logs/scene_annos.txt"
    )
    s3d_scene_line_err = os.path.join(
        s3d_project_pth, "logs/scene_line_err.txt"
    )

    # 观测数据缺失
    s3d_pano_obs_err = os.path.join(
        s3d_project_pth, "logs/scene_observation.txt"
    ) # 全景图
    s3d_pesp_obs_err = os.path.join(
        s3d_project_pth, "logs/pesp_obs_loss.txt"
    ) # 单视角视图

    # 数据集目录
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

# 全景图观测数据缺失的场景
with open(s3d_pano_obs_err, encoding="utf-8") as f:
    s3d_pano_invalid = f.readlines()
for index, item in enumerate(s3d_pano_invalid):
    s3d_pano_invalid[index] = item.replace("\n", "")

# 单视角视图观测数据缺失的场景
with open(s3d_pesp_obs_err, encoding='utf-8') as f:
    s3d_pesp_invalid = f.readlines()
for index, item in enumerate(s3d_pesp_invalid):
    s3d_pesp_invalid[index] = item.replace("\n", "")
