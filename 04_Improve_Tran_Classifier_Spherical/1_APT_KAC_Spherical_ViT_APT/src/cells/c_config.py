# ==============================================================================
# 1. CẤU HÌNH
# ==============================================================================
DATASETS = ["CUB200"]      # chọn trong ["CIFAR100", "CUB200", "StanfordCars", "ImageNet_R"] (nên 1 dataset mỗi phiên)
PHASE    = "val"           # "val": học 80% Train, đánh giá 20% Val -> chọn cấu hình (KHÔNG đụng Test)
                           # "test": học 100% Train, đánh giá Test với cấu hình đã chọn ở "val" (cần best_configs_val.json)
SEEDS    = [0, 1, 2]       # seed s: thứ tự task = RandomState(s).permutation, Val = Fold[s] (quy trình của nhóm)
HEADS    = ["kac"]
KAC_GRIDS = 4              # paper KAC: N = 4 RBF
HEAD_LR  = 0.001           # lr riêng của head KAC
APT_DUMP_APT = "2"         # "2": lưu đặc trưng ViT đóng băng + APT (frozen, apt_stale, apt_refresh); "1": bỏ apt_refresh (nhanh hơn)
BATCH, PROMPT_PARAM = 64, 0.01
# Siêu tham số APT lấy từ experiments/*.sh của repo chính thức
HP = {"CIFAR100":     dict(lr=0.004, epochs=30, ema=0.7, config="configs/cifar-100_prompt.yaml"),
      "CUB200":       dict(lr=0.02,  epochs=25, ema=0.7, config="configs/cub200_prompt.yaml"),
      "StanfordCars": dict(lr=0.005, epochs=50, ema=0.7, config="configs/stanfordcars_prompt.yaml"),
      "ImageNet_R":   dict(lr=0.003, epochs=30, ema=0.8, config="configs/imnet-r_prompt.yaml")}
DRIVE_DIR = "/kaggle/working/apt_spherical"
ZIP_PATH  = None           # tự tìm repo APT trong /kaggle/input
assert PHASE in ("val", "test") and set(SEEDS) <= {0, 1, 2, 3, 4} and set(DATASETS) <= set(HP)
