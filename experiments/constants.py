import os

MAIN_SEED = 93

BASE_PATH = "<BASE_PATH_TO_DATA_DIRECTORIES>"

PATH_DATA_LARGE_P = lambda scaled_string, p, N, additive, graph_type: os.path.join(BASE_PATH, "large_p", scaled_string, f"graph_{p}_{N}_additive_{additive}_{graph_type}")
PATH_DATA_SMALL_P = lambda N, scaled_string, p, additive, graph_type: os.path.join(BASE_PATH, "small_p", str(N), scaled_string, f"graph_{p}_{N}_additive_{additive}_{graph_type}")