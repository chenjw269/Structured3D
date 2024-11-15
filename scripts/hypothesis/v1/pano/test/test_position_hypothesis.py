import sys
sys.path.append(".")

from s3d import *

from scripts.hypothesis.v1.pano.position_hypothesis import *


if __name__ == "__main__":

    scene_index = "scene_02600"
    
    # 场景边界数据
    scene_info = os.path.join(s3d_data_pth, scene_index, "boundary.csv")
    scene_info = pd.read_csv(scene_info)
    # 从场景中采样位置假设
    hypothesis_loc = generate_scene_hypothesis(scene_info)
    print(f"{hypothesis_loc.shape[0]} positions")

    # 场景地图
    scene_map = os.path.join(s3d_map_pth, scene_index, "map.png")
    scene_map = cv2.imread(scene_map)
    # 可视化位置假设
    for i in hypothesis_loc:
        i = i.tolist()
        scene_map = cv2.circle(scene_map, i, 1, (0,0,255), -1)
    cv2.imshow("Visualize hypothesis", scene_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()