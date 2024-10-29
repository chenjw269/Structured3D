import random
import pandas as pd


class S3DPespMC(Dataset):
    
    def __init__(self, csv_pth):
        super(S3DMetricLearning, self).__init__()
        
        self.data = pd.read_csv(csv_pth)
        self.resolution = 25 # 2.5 cm, 0.025 m / pixel

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        # 局部地图 Lc,Lc
        local_map = self.data['local map'][index]
        local_map = torch.Tensor(np.load(local_map))
        # 局部地图加上受限视野
        
        # 局部地图加上虚拟旋转角
        rotation = random.randint(0, 360)
        
        # 全局地图 Lg,Lg
        global_map = self.data['global map'][index]
        global_map = torch.Tensor(np.load(global_map))
        # 真实位置 2
        gt_pos = eval(self.data['gt pos'][index])
        gt_pos = torch.Tensor(gt_pos)
        
        # 当前场景标注
        scene_annos = self.data['annos'][index]
        scene_annos_df = pd.read_csv(scene_annos)
        # 场景坐标范围
        scene_bound = np.array([
            [128, int(scene_annos_df['size_x'].item() / self.resolution) - 128],
            [128, int(scene_annos_df['size_y'].item() / self.resolution) - 128]
        ])
        
        ########################
        # 随机正样例
        ########################
        positive_nums = 5
        # 随机正样例位置
        positive_radius = 60 # 60 * 0.025 = 1.5
        positive_sample_pos = generate_neighbor_within(
            gt_pos, positive_nums, positive_radius, scene_bound
        )
        positive_lm = extract_local_patches(global_map, positive_sample_pos)
        # 随机正样例朝向
        positive_angle = 15
        random.uniform(rotation - positive_angle, rotation + positive_angle)
        positive_lm = 
        
        positive_lm = torch.Tensor(positive_lm)
        
        ########################
        # 随机正样例
        ########################
        
        # 位置相近，但角度不同的负样例
        
        # 位置相远，但角度相近的负样例
        
        