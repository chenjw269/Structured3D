def virtual_pano_obs(map_image, pose, view_range=256):
    
    # 获取地图上，对应位置周围的部分
    virtual_pano_bev = map_image[
        int(pose[0] - view_range/2) : int(pose[0] + view_range/2),
        int(pose[1] - view_range/2) : int(pose[1] + view_range/2),
        :
    ]

    # 考虑从自身位置出发的可见性
    
    
    return virtual_pano_bev
