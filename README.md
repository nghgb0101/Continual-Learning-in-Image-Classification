# 📦 Continual Learning in Image Classification — Group 3

**Dự án:** Continual Learning with Pre-trained Vision Models: A Comprehensive Study on Parameter-Efficient Adaptation and Classifier Architectures  
**Giảng viên hướng dẫn:** TS. Phan Thị Thu Hồng  
**Nhóm sinh viên thực hiện:**  
1. **Trịnh Ngọc Thảo Hiền** — Nhánh 2: Hướng giải nghiệm đóng giải tích (Closed-Form Ridge Solver & SO-KAC)
2. **Bảo** — Nhánh 3: Hướng cải tiến Backbone APT
3. **Trân** — Nhánh 4: Hướng cải tiến Classifier Spherical (Mặt cầu siêu đa tạp)

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
├── 02_Improve_Hien_Nghiem_Dong/                    <-- [NHÁNH 2: CẢI TIẾN CỦA HIỀN - GIẢI NGHIỆM ĐÓNG]
│   └── .gitkeep                                    <-- (Thư mục chờ nạp mã nguồn hoàn chỉnh của Hiền)
│
├── 03_Improve_Bao_Backbone_APT/                    <-- [NHÁNH 3: CẢI TIẾN CỦA BẢO - BACKBONE APT]
│   └── .gitkeep                                    <-- (Thư mục chờ nạp mã nguồn hoàn chỉnh của Bảo)
│
└── 04_Improve_Tran_Classifier_Spherical/           <-- [NHÁNH 4: CẢI TIẾN CỦA TRÂN - CLASSIFIER SPHERICAL]
    └── .gitkeep                                    <-- (Thư mục chờ nạp mã nguồn hoàn chỉnh của Trân)
```

---

## ⚙️ Hướng Dẫn Cài Đặt Môi Trường

Khuyến nghị sử dụng Python 3.10+ trên môi trường GPU (CUDA 11.8 hoặc 12.x):
```bash
pip install -r requirements.txt
```
