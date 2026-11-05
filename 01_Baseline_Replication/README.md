# 01. Baseline Replication Suite (M0 vs M1)

Thư mục này chứa mã nguồn và các Jupyter Notebook tái lập (Replication) độc lập cho **4 phương pháp Continual Prompt Tuning SOTA** trên bộ dữ liệu chuẩn **CUB-200-2011** (10 tasks x 20 classes):

---

## 📑 Danh Sách Notebooks Thực Nghiệm

| File Notebook | Mô hình & Bài báo gốc | Chế độ khảo sát | Mô tả chức năng |
| :--- | :--- | :---: | :--- |
| **`Replication_L2P_CUB200_M0_M1.ipynb`** | **L2P** (Wang et al., CVPR 2022) | **M0 vs M1** | Khảo sát cơ chế Prompt Pool của L2P với Linear Head gốc (M0) vs Naive KAC Head (M1). |
| **`Replication_DualPrompt_CUB200_M0_M1.ipynb`** | **DualPrompt** (Wang et al., ECCV 2022) | **M0 vs M1** | Khảo sát kiến trúc G-Prompt & E-Prompt với Linear Head gốc (M0) vs Naive KAC Head (M1). |
| **`Replication_CODA_Prompt_CUB200_M0_M1.ipynb`** | **CODA-Prompt** (Smith et al., CVPR 2023) | **M0 vs M1** | Khảo sát cơ chế Attention Decomposed Prompts với Linear Head gốc (M0) vs Naive KAC Head (M1). |
| **`Replication_APT_CUB200_M0_M1.ipynb`** | **APT** (Chen et al., ICCV 2025) | **M0 vs M1** | Khảo sát Additive Prompt kết hợp EMA Merge với Cosine Head gốc (M0) vs Naive KAC Head (M1). |

---

## 🔬 Định Nghĩa Hai Chế Độ Đối Chứng

- **M0 (Paper Baseline):** Sử dụng 100% kiến trúc đầu phân loại (Linear/Cosine Head) và thiết lập nguyên bản từ bài báo tác giả gốc.
- **M1 (Naive KAC):** Thay thế đầu phân loại bằng **KAC Layer** nguyên bản (Hu et al., CVPR 2025) gồm:
  $$\mathbf{x}_{\text{norm}} = \text{LayerNorm}(\mathbf{f}), \quad \phi_k(x) = \exp\left(-\left(\frac{x - c_k}{\Delta}\right)^2\right), \quad \hat{\mathbf{y}} = \mathbf{W}_{\text{spline}} \cdot \phi(\mathbf{x}_{\text{norm}})$$
  với $G = 4$ grids trong khoảng $[-2.0, 2.0]$.

---

## 🚀 Hướng Dẫn Chạy Nhanh (1-Click Run)

Tất cả các notebook đều được cấu hình tự động nhận diện môi trường:
- **Trên Kaggle:** Mở notebook, kết nối GPU T4, nạp input CUB-200 và bấm **"Run All"**.
- **Trên Local / GPU Server:** Cài đặt thư viện theo `../requirements.txt` và chạy tuần tự các cell.
