# 02. Hướng Nghiên Cứu Cải Tiến: Giải Nghiệm Đóng (Closed-Form Ridge Solver & SO-KAC)

**Thành viên phụ trách:** Trịnh Ngọc Thảo Hiền  
**Đề tài:** Kolmogorov-Arnold Classifiers with Closed-Form Ridge Solver for Continual Learning  
**Trọng tâm:** Chuyển dịch toàn diện từ tối ưu hóa lặp SGD sang **Nghiệm đóng giải tích Ridge Regression** kết hợp khai triển hàm phi tuyến bậc hai **SO-KAC (Second-Order Kolmogorov-Arnold Classifier)** trên các kiến trúc Vision Transformer (ViT-B/16 và Swin Transformer).

---

## 🔬 Cơ Sở Lý Thuyết & Toán Học

### 1. Nguyên Lý Tích Lũy Thống Kê Bậc Hai & Nghiệm Đóng
Thay vì cập nhật trọng số phân loại lặp đi lặp lại qua các epoch gây trôi dạt và quên lãng (catastrophic forgetting), phương pháp lưu trữ ma trận hiệp phương sai tích lũy không thiên vị qua từng task $t \in \{1, \dots, T\}$:

$$
\mathbf{\Sigma} = \sum_{t=1}^T \mathbf{X}_t^\top \mathbf{X}_t \in \mathbb{R}^{D \times D}, \quad \mathbf{P} = \sum_{t=1}^T \mathbf{X}_t^\top \mathbf{Y}_t \in \mathbb{R}^{D \times C}
$$

Nghiệm tối ưu toàn cục được giải trực tiếp trong **1.27 giây** bằng phân rã Cholesky độ chính xác kép (Float64):

$$
\mathbf{W}^* = (\mathbf{\Sigma} + \lambda \mathbf{I})^{-1} \mathbf{P}
$$

### 2. Khai Triển ANOVA Bậc Hai & Trực Giao Haar (SO-KAC)
Để mở rộng năng lực biểu diễn phi tuyến mà không làm bùng nổ số chiều như phép chiếu ngẫu nhiên thuần túy (RanPAC $M = 10,000$), SO-KAC kết hợp thành phần tuyến tính trực tiếp và tương tác bậc hai qua phép chiếu trực giao Haar:

$$
\Psi(\mathbf{x}) = [\mathbf{x}, \, \phi(\mathbf{Q}_{\text{orth}} \mathbf{x})]
$$

---

## 📑 Danh Mục Notebooks Thực Nghiệm (Tất cả đều chạy liên hoàn 4 Datasets)

### 🌟 1. Main Benchmarks (Bảng Kết Quả Chính Đối Chứng Bài Báo)
| STT | File Notebook | Mục tiêu khoa học |
| :---: | :--- | :--- |
| **1** | **`1_Benchmark_RanPAC_Replication_4Datasets.ipynb`** | Tái lập nguyên bản 100% thuật toán RanPAC (NeurIPS 2023, $M=10,000$) trên cả 4 datasets (CIFAR-100, CUB-200, Cars-196, ImageNet-R). |
| **2** | **`2_Benchmark_ViT_SO_KAC_4Datasets.ipynb`** | Đánh giá nghiệm đóng SO-KAC trên nền tảng ViT-B/16 đông cứng qua 4 datasets, giải quyết hiện tượng sập nghiệm của RanPAC. |
| **3** | **`3_Proposed_Swin_SO_KAC_4Datasets.ipynb`** | **Mô hình đề xuất chính thức SOTA:** Kết hợp Swin Transformer phân cấp + ConvPass PEFT + Stage 4 GAP (768d) + SO-KAC v5 Ridge Solver. |

---

### 🔬 2. Ablations & Scientific Analysis (Khảo Sát Bóc Tách Thành Phần Chuyên Sâu)
| STT | File Notebook | Trục khảo sát bóc tách |
| :---: | :--- | :--- |
| **1** | **`1_Ablation_Classifier_SO_KAC.ipynb`** | **Bóc tách Classifier Head:** Cố định backbone ViT APT, khảo sát 7 nhóm / 31 cấu hình KAC (Knots ReLU vs Spline vs RBF, số lượng Heads 1..8, ma trận Haar QR vs Gaussian, Alignment). |
| **2** | **`2_Ablation_Backbone_Swin_Evolution.ipynb`** | **Bóc tách Backbone Swin:** Cố định Head KAC, khảo sát chuỗi tiến hóa 6 bước từ Linear sang SO-KAC, đối chứng F3 Multistage (1152d) vs GAP Stage 4 (768d). |
| **3** | **`3_Ablation_ViT_Weights_IN21K_vs_IN1K.ipynb`** | **Khảo sát Nền tảng ViT:** So sánh trực diện biểu diễn của ViT đông cứng (Zero-PEFT) trên 2 bộ trọng số tiền huấn luyện ImageNet-21k vs ImageNet-1k. |
| **4** | **`4_Ablation_Swin_Scales_Tiny_vs_Small.ipynb`** | **Khảo sát Nền tảng Swin:** Đánh giá ảnh hưởng của quy mô mô hình phân cấp giữa Swin-Tiny (28M params) vs Swin-Small (50M params) và cơ chế pooling Top-Down FPN. |
| **5** | **`5_Tuning_Hyperparameters_Suite.ipynb`** | **Khảo sát Siêu tham số:** Quét đơn biến toàn diện trên 11 trục độc lập (learning rate, weight decay, batch size, warmup, scale, rank, scheduler, ema) đo chất lượng hình học bằng NCM. |

---

## 🚀 Hướng Dẫn Chạy Nhanh (1-Click Run)

Tất cả 8 notebook đều được thiết kế **tự chứa (self-contained)** và có sẵn bộ tự động nhận diện môi trường:
- **Kaggle / GPU Server:** Mở bất kỳ notebook nào, bật GPU (T4 hoặc P100/A100) và chọn **"Run All"**.
- Code tự động nạp dữ liệu, huấn luyện, giải nghiệm đóng và in bảng tổng hợp Last Accuracy, Average Incremental Accuracy, Task-Aware Accuracy và Effective Rank.
