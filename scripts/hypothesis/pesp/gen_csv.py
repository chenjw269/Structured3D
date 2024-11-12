def generate_scene_csv(scene_index_list):
    
    for scene_index in scene_index_list:
        
        # 场景地图
        scene_map = os.path.join(s3d_map_pth, scene_index, "map.npy")
        # 场景边界
        scene_bound = pd.read_csv(os.path.join(s3d_data_pth, scene_index, "boundary.csv"))
