# ==============================================================================
# 1. CẤU HÌNH
# ==============================================================================
DATASETS = ["CIFAR100", "CUB200", "StanfordCars", "ImageNet_R"]   # cả 4 trong một phiên (không train)
SEEDS    = [0, 1, 2]
PHASE    = "val"            # "val": tâm lớp từ 80% Train, đánh giá trên 20% Val (KHÔNG đụng Test)
# backbone đóng băng cho nhánh Spherical: {tên timm: nhãn trong bảng}
BACKBONES = {"swin_base_patch4_window7_224.ms_in22k_ft_in1k": "Swin-B",    # ~88M tham số, cùng cỡ ViT-B/16
             "swin_tiny_patch4_window7_224.ms_in1k":          "Swin-T"}    # ~28M tham số, nhẹ hơn ~3 lần
BATCH, WORKERS = 128, 4
OUT = "/kaggle/working/swin_spherical"
INPUT_ROOT = "/kaggle/input"
PRETRAINED = True           # chỉ để False khi chạy thử không có Internet
import os; os.makedirs(OUT, exist_ok=True)
assert PHASE in ("val", "test") and set(SEEDS) <= {0, 1, 2, 3, 4}
