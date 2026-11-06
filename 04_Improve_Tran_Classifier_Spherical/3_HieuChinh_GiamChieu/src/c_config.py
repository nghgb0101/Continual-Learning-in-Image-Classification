# ==============================================================================
# 1. CẤU HÌNH  (chỉ đọc dump có sẵn -> chạy CPU, không cần GPU / Internet)
# ==============================================================================
DATASETS = ["CIFAR100", "CUB200", "StanfordCars", "ImageNet_R"]
SEEDS    = [0, 1, 2]
PHASE    = "val"
INPUT_ROOT = "/kaggle/input"
OUT = "/kaggle/working/exp_calib_proj"
# Thử nghiệm 1: hiệu chỉnh logit KAC theo từng khối task (0 tham số, 0 nhánh phụ)
CALIB = ["raw", "center", "center_half", "zscore", "block_softmax"]
CALIB_FUSE = ["center", "block_softmax"]      # các cách hiệu chỉnh đem ghép tiếp với Spherical-ViT (fuse/hier)
# Thử nghiệm 2: giảm chiều đặc trưng Spherical (ViT đóng băng) 768 -> d
PROJ_DIMS  = [64, 128, 256]
PROJ_TYPES = ["pca", "rp"]                    # pca: học từ ảnh train task 1 rồi đóng băng; rp: chiếu ngẫu nhiên trực giao, không cần dữ liệu
import os; os.makedirs(OUT, exist_ok=True)
