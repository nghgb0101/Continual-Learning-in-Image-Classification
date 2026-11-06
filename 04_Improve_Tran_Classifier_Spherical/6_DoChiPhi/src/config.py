# ==============================================================================
# 1. CẤU HÌNH: đo chi phí tính toán của APT + KAC và nhánh Spherical (4 dataset)
#    Không train đầy đủ: mỗi dataset chỉ train Task 1 trong vài epoch để đo tốc độ, rồi ngoại suy.
# ==============================================================================
DATASETS = ["CIFAR100", "CUB200", "StanfordCars", "ImageNet_R"]
PHASE = "val"
SEED = 0                      # dùng thứ tự lớp + split Val của seed 0 (quy trình của nhóm)
PROFILE_EPOCHS = 2            # số epoch train Task 1 để đo (epoch đầu có warm-up, lấy số đo ở epoch cuối)
BATCH_TRAIN = 64              # = repo APT chính thức
BATCH_INFER = 64
WORKERS = 4                   # Kaggle có 4 CPU
K_CENTRES = 5                 # số tâm mỗi lớp (giá trị lớn nhất trong lưới -> chi phí lớn nhất)
N_WARM, N_REP = 3, 10         # số lần chạy nóng máy / số lần đo cho phần đo GPU thuần
# Siêu tham số APT theo experiments/*.sh của repo (chỉ dùng để ngoại suy thời gian train đầy đủ)
FULL_EPOCHS = {"CIFAR100": 30, "CUB200": 25, "StanfordCars": 50, "ImageNet_R": 30}
PROMPT_LR   = {"CIFAR100": 0.004, "CUB200": 0.02, "StanfordCars": 0.005, "ImageNet_R": 0.003}
HEAD_LR = 1e-3
VIT_NAME = "vit_base_patch16_224"
SWIN = {"swin_base_patch4_window7_224.ms_in22k_ft_in1k": "Swin-B",
        "swin_tiny_patch4_window7_224.ms_in1k": "Swin-T"}
# Trọng số không ảnh hưởng tới thời gian / bộ nhớ / FLOPs -> mặc định KHÔNG tải (không cần Internet).
PRETRAINED = False
INPUT_ROOT = "/kaggle/input"
OUT = "/kaggle/working/cost_profile"
SMOKE = False                 # True: chạy thử cực nhanh (chỉ để kiểm tra code)
import os; os.makedirs(OUT, exist_ok=True)
if SMOKE:
    PROFILE_EPOCHS, BATCH_TRAIN, BATCH_INFER, N_WARM, N_REP, WORKERS = 1, 4, 4, 1, 1, 0
