import sys
sys.path.append(".")

from scripts.bev_observation.bev_processing_mp import execute_bev_processing


task = "scene_02785/2D_rendering/141"

execute_bev_processing(task)
