# 📦 Continual Learning in Image Classification — Group 3

**Dự án:** Continual Learning with Pre-trained Vision Models: A Comprehensive Study on Parameter-Efficient Adaptation and Classifier Architectures  
**Giảng viên hướng dẫn:** PGS. TS. Phan Thị Thu Hồng  
**Nhóm sinh viên thực hiện:**  
1. **Trịnh Ngọc Thảo Hiền** — Nhánh 2: Hướng giải nghiệm đóng giải tích (Closed-Form Ridge Solver & SO-KAC)
2. **Nguyễn Hữu Gia Bảo** — Nhánh 3: Hướng cải tiến Backbone APT
3. **Huỳnh Thị Nhã Trân** — Nhánh 4: Hướng cải tiến Classifier Spherical (Mặt cầu siêu đa tạp)

---

## 🗂️ Cấu Trúc Mã Nguồn (4 Nhánh Nghiên Cứu)

```text
.
├── README.md                                       <-- (Tài liệu này) Tổng quan cấu trúc và hướng dẫn
├── requirements.txt                                <-- Danh mục thư viện Python môi trường chuẩn
│
├── 01_Baseline_Replication/                        <-- [NHÁNH 1: REPLICATION NỀN TẢNG (CHUNG CẢ NHÓM)]
│   ├── README.md                                   <-- Báo cáo chi tiết kịch bản Multi-Split (5T, 10T, 20T, 40T)
│   ├── Replication_L2P_CUB200_M0_M1.ipynb          <-- L2P (CVPR 2022): M0 Linear vs M1 KAC
│   ├── Replication_DualPrompt_CUB200_M0_M1.ipynb   <-- DualPrompt (ECCV 2022): M0 Linear vs M1 KAC
│   ├── Replication_CODA_Prompt_CUB200_M0_M1.ipynb  <-- CODA-Prompt (CVPR 2023): M0 Linear vs M1 KAC
│   └── Replication_APT_CUB200_M0_M1.ipynb          <-- APT (ICCV 2025): M0 Cosine vs M1 KAC
│
├── 02_Improve_Hien_Nghiem_Dong/                    <-- [NHÁNH 2: CẢI TIẾN CỦA THẢO HIỀN - GIẢI NGHIỆM ĐÓNG]
│   ├── README.md                                   <-- Cơ sở toán học & Bảng ánh xạ thực nghiệm vào Report
│   ├── 1_Main_Benchmarks/                          <-- [3 Mốc đối chứng lớn nhất bài báo (4 Datasets)]
│   │   ├── 1_Benchmark_RanPAC_Replication_4Datasets.ipynb
│   │   ├── 2_Benchmark_ViT_SO_KAC_4Datasets.ipynb
│   │   └── 3_Proposed_Swin_SO_KAC_4Datasets.ipynb
│   └── 2_Ablations_and_Analysis/                   <-- [5 Bài bóc tách thành phần chuyên sâu (4 Datasets)]
│       ├── 1_Ablation_Classifier_SO_KAC.ipynb
│       ├── 2_Ablation_Backbone_Swin_Evolution.ipynb
│       ├── 3_Ablation_ViT_Weights_IN21K_vs_IN1K.ipynb
│       ├── 4_Ablation_Swin_Scales_Tiny_vs_Small.ipynb
│       └── 5_Tuning_Hyperparameters_Suite.ipynb
│
├── 03_Improve_Bao_Backbone_APT/                    <-- [NHÁNH 3: CẢI TIẾN CỦA GIA BẢO - BACKBONE APT]
│   └── .gitkeep                                    <-- (Thư mục chờ nạp mã nguồn hoàn chỉnh của Gia Bảo)
│
└── 04_Improve_Tran_Classifier_Spherical/           <-- [NHÁNH 4: CẢI TIẾN CỦA NHÃ TRÂN - CLASSIFIER SPHERICAL]
    └── .gitkeep                                    <-- (Thư mục chờ nạp mã nguồn hoàn chỉnh của Nhã Trân)
```

---

## 📊 Bộ Dữ Liệu Thực Nghiệm Trên Kaggle (Datasets)

Để thuận tiện cho việc tái lập kết quả 1-Click trên môi trường Kaggle GPU, toàn bộ 4 tập dữ liệu chuẩn đã được định dạng và chia sẵn `train/test`:

| Tập Dữ Liệu | Thể Loại Phân Loại | Quy Mô & Kịch Bản Task | Liên Kết Kaggle Dataset |
| :--- | :--- | :---: | :--- |
| **CIFAR-100** | Phân loại vật thể tổng quát | 100 lớp (10 tasks x 10 lớp) | [thohintrnhngc/cifar-100-python](https://www.kaggle.com/datasets/thohintrnhngc/cifar-100-python) |
| **CUB-200-2011** | Phân loại loài chim fine-grained | 200 lớp (10 tasks x 20 lớp) | [cyizhuo/cub-200-2011-by-classes-folder](https://www.kaggle.com/datasets/cyizhuo/cub-200-2011-by-classes-folder) |
| **Stanford Cars-196** | Phân loại xe ô tô fine-grained | 196 lớp (9x20 + 1x16 lớp) | [jutrera/stanford-car-dataset-by-classes-folder](https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder) |
| **ImageNet-R** | Dữ liệu lệch miền nghệ thuật (OOD) | 200 lớp (10 tasks x 20 lớp) | [thohintrnhngc/imagenet-r](https://www.kaggle.com/datasets/thohintrnhngc/imagenet-r) |

---

## ⚙️ Hướng Dẫn Cài Đặt Môi Trường

Khuyến nghị sử dụng Python 3.10+ trên môi trường GPU (CUDA 11.8 hoặc 12.x):
```bash
pip install -r requirements.txt
```
