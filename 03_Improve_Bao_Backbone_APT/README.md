# 03. Hướng Nghiên Cứu Cải Tiến: Backbone APT — Hierarchical Layer-Wise PPF (L-PPF) & Multi-Depth Residual Adapters

> **Dự án:** Continual Learning with Pre-trained Vision Models: A Comprehensive Study on Parameter-Efficient Adaptation and Classifier Architectures  
> **Môn học:** Deep Learning Project (DPL302m)  
> **Giảng viên hướng dẫn:** PGS. TS. Phan Thị Thu Hồng

---

## 📌 1. Đặt Vấn Đề & Cơ Sở Lý Thuyết (Theoretical Motivation)

Trong bài toán **Exemplar-Free Class-Incremental Learning (EFCIL)**, mô hình phải học tuần tự các tập lớp mới mà tuyệt đối **không lưu trữ mẫu cũ** ($M_{\text{buffer}} = 0$) và **không biết ID của task khi suy luận**.

Mô hình nền tảng **Additive Prompt Tuning (APT - ICCV 2025)** đóng băng toàn bộ backbone Vision Transformer (ViT-B/16, 85.8M tham số) và tối ưu một tập prompt cộng tính $\Phi_t = \{\mathbf{p}_{k}^l, \mathbf{p}_{v}^l\}_{l=1}^{12}$ tại vector [CLS]. Quá trình chuyển giao tri thức giữa các task được thực hiện qua cơ chế **Progressive Prompt Fusion (PPF)**:

$$\mathbf{P}_t^{\text{PPF}} = \alpha \mathbf{P}_{t-1}^{\text{PPF}} + (1 - \alpha) \mathbf{P}_t$$

Qua quá trình tái lập và kiểm chứng nghiêm ngặt, chúng tôi phát hiện **2 điểm nghẽn cốt tử** ở giai đoạn thích ứng backbone (Stage I):

```
                   CÁC ĐIỂM NGHẼN CỐT TỬ CỦA APT TRUYỀN THỐNG
┌───────────────────────────────────────────────┐ ┌───────────────────────────────────────────────┐
│     1. Uniform EMA Rigidity (Độ cứng ngắc)   │ │      2. Linear Prompt Expressivity Limits     │
│   • Áp đặt một hệ số duy nhất α = 0.70 cho    │ │   • Prompt chỉ cộng tuyến tính vào Key/Value  │
│     toàn bộ 12 tầng Transformer.              │ │     của token [CLS].                          │
│   • Vi phạm tính phân cấp đặc trưng của ViT:  │ │   • Bất lực trong việc nắn chỉnh phi tuyến    │
│     Tầng nông cần giữ ổn định (stability),    │ │     đa tạp đặc trưng tại mạng FFN khi phân    │
│     tầng sâu cần linh hoạt thích ứng          │ │     loại các lớp fine-grained tương đồng.     │
│     (plasticity).                             │ │                                               │
└───────────────────────────────────────────────┘ └───────────────────────────────────────────────┘
```

---

## 🔬 2. Các Đóng Góp Cải Tiến Đề Xuất (Proposed Solutions)

### 2.1. Hierarchical Layer-Wise PPF (L-PPF) & Reverse L-PPF
Thay vì một giá trị $\alpha$ cố định, mỗi tầng Transformer $l \in \{1, \dots, L\}$ ($L=12$) được gán một hệ số lưu giữ riêng $\alpha_l \in [0, 1]$ biến thiên tuyến tính theo độ sâu:

$$\alpha_l = \alpha_{\text{shallow}} - \frac{l - 1}{L - 1} (\alpha_{\text{shallow}} - \alpha_{\text{deep}})$$

Để đảm bảo đối chứng công bằng (apple-to-apple comparison) với APT gốc, tổng ngân sách duy trì tham số trung bình được bảo toàn tuyệt đối:

$$\bar{\alpha} = \frac{1}{L} \sum_{l=1}^L \alpha_l = \frac{\alpha_{\text{shallow}} + \alpha_{\text{deep}}}{2} = 0.70$$

* **Canonical L-PPF ($\alpha_l: 0.85 \to 0.55$):**
  * **Tầng nông ($l \le 4$, $\alpha_l \approx 0.85$):** Neo giữ vững chắc các bộ trích xuất hình học không gian sơ cấp (edges, textures), chống trôi dạt biểu diễn nền tảng.
  * **Tầng sâu ($l \ge 9$, $\alpha_l \approx 0.55$):** Tăng cường tính dẻo dai ($1 - \alpha_l = 0.45$), cho phép hấp thụ nhanh chóng ranh giới quyết định của các lớp mới.
* **Reverse L-PPF ($\alpha_l: 0.55 \to 0.85$ - Negative Control):**
  * Mô hình đối chứng phủ định nhằm thử nghiệm: nếu nới lỏng tầng nông và khóa cứng tầng sâu thì điều gì xảy ra.

---

### 2.2. Multi-Depth Residual Bottleneck Adapters
Giải quyết giới hạn biểu diễn tuyến tính bằng cách ghép song song nhánh bottleneck adaptee vào khối FFN tại tầng $l$:

$$\mathbf{h}_{\text{adapt}}^{(l)} = \sigma\big(\mathbf{u} \mathbf{W}_{\text{down}}^{(l)} + \mathbf{b}_{\text{down}}^{(l)}\big) \mathbf{W}_{\text{up}}^{(l)} + \mathbf{b}_{\text{up}}^{(l)}$$
$$\mathbf{x}_{\text{out}}^{(l)} = \mathbf{x} + \mathrm{MLP}(\mathbf{u}) + s \cdot \mathbf{h}_{\text{adapt}}^{(l)}$$

* $\mathbf{W}_{\text{down}} \in \mathbb{R}^{D \times r}$, $\mathbf{W}_{\text{up}} \in \mathbb{R}^{r \times D}$ với hạng nút thắt $r = 16 \ll D = 768$, hàm kích hoạt $\text{GELU}$, và hệ số co giãn $s = 0.1$.
* **Điều kiện khởi tạo Zero-Init:** $\mathbf{W}_{\text{up}} = \mathbf{0}$, $\mathbf{b}_{\text{up}} = \mathbf{0}$. Đảm bảo tại bước $0$, $\mathbf{h}_{\text{adapt}}^{(l)} \equiv \mathbf{0}$, loại bỏ hoàn toàn hiện tượng sốc khởi tạo (initialization shock) lên biểu diễn tiền huấn luyện.
* **Khảo sát 3 không gian tích hợp chiều sâu ($\Omega_{\text{adapt}}$):**
  1. **Full-Depth (Tầng 0–11, 12 layers, +0.304M params):** Nâng cấp năng lực phi tuyến toàn diện.
  2. **Deep-Only (Tầng 6–11, 6 layers, +0.152M params):** Tiết kiệm 50% tham số adapter, giữ đóng băng 100% các tầng sơ cấp.
  3. **Top-3 Head Scope (Tầng 9–11, 3 layers, +0.076M params):** Tiết kiệm 75% tham số, đóng vai trò như bộ chiếu phi tuyến sát đầu phân loại.

---

### 2.3. Dual-Memory & Unified Inference
Sau khi kết thúc task $t$, cả trọng số prompt và adapter đều được hợp nhất qua hệ số phân tầng $\alpha_l$:

$$\mathbf{P}_t^{\text{PPF}, (l)} = \alpha_l \mathbf{P}_{t-1}^{\text{PPF}, (l)} + (1 - \alpha_l) \mathbf{P}_t^{(l)}, \quad \mathbf{W}_t^{\text{PPF}, (l)} = \alpha_l \mathbf{W}_{t-1}^{\text{PPF}, (l)} + (1 - \alpha_l) \mathbf{W}_t^{(l)}$$

Tại thời điểm suy luận, mẫu kiểm tra $x$ được xử lý trực tiếp qua mạng hợp nhất mà **không cần task ID**, kết hợp đầu phân loại **Hybrid Head (Detach KAC Gradient)** để đưa ra dự đoán tối ưu:

   $$\hat{y} = \arg\max_{c \in \mathcal{C}_{1:t}} \left[ \text{softmax}\left(\frac{\ell_{\text{cos}}(x)}{\tau}\right) + \alpha_{\text{hyb}} \text{softmax}(\ell_{\text{kac}}(x)) \right]_c$$

---

## 🗂️ 3. Cấu Trúc Mã Nguồn & Danh Mục Thực Nghiệm

```text
My_Part/ (03_Improve_Bao_Backbone_APT)
├── README.md                                                        <-- (Tài liệu này) Báo cáo chi tiết phương pháp & kết quả
│
├── CUB200_Experiments/                             <-- [Bộ thí nghiệm chuyên sâu Fine-grained (10 Notebooks CUB-200)]
│   ├── apt_cub200_kac_classifier_val_split.ipynb                    <-- [BASELINE] APT + KAC Classifier (Uniform PPF, α=0.70)
│   ├── apt_cub200_hybrid_classifier_val_split.ipynb                 <-- [CẢI TIẾN CLASSIFIER] APT + Hybrid Head (Cosine + Detach KAC)
│   ├── apt_cub200_lppf_val_split.ipynb                              <-- [CẢI TIẾN BACKBONE] APT + Hierarchical L-PPF (α: 0.85 → 0.55)
│   ├── apt_cub200_reverse_lppf_val_split.ipynb                      <-- [ABLATION ĐỐI CHỨNG] APT + Reverse L-PPF (α: 0.55 → 0.85)
│   ├── apt_cub200_adapter_uniform-ppf_val_split.ipynb               <-- [CẢI TIẾN ADAPTER] Full Adapter (0–11) + Uniform PPF
│   ├── apt_cub200_adapter_lppf_val_split.ipynb                      <-- [CẢI TIẾN KẾT HỢP] Full Adapter (0–11) + L-PPF 
│   ├── apt_cub200_adapter_deep6_11_uniform-ppf_val_split.ipynb      <-- [DEEP ADAPTER] Adapter tầng sâu (6–11) + Uniform PPF
│   ├── apt_cub200_adapter_deep6_11_lppf_val_split.ipynb             <-- [DEEP ADAPTER] Adapter tầng sâu (6–11) + L-PPF 
│   ├── apt_cub200_adapter_deep9_11_uniform-ppf_val_split.ipynb      <-- [TOP-3 ADAPTER] Adapter tầng sâu (9–11) + Uniform PPF
│   └── apt_cub200_adapter_deep9_11_lppf_val_split.ipynb             <-- [TOP-3 ADAPTER] Adapter tầng sâu (9–11) + L-PPF
│
├── Multi_Dataset_Validation(CIFAR100 & STANDFORD CARS & ImageNet-R)/                                      <-- [Bộ thẩm định đa miền dữ liệu (10 Notebooks 3 Datasets)]
│   ├── apt_3dataset_kac_val_split.ipynb                             <-- [BASELINE] APT + KAC Classifier trên 3 Datasets
│   ├── apt_3dataset_hybrid_val_split_1.ipynb                        <-- APT + Hybrid Head trên 3 Datasets
│   ├── apt_3dataset_ppf_upgrade_val_split_1.ipynb                   <-- APT + L-PPF trên 3 Datasets
│   ├── apt_3dataset_reverse_lppf_val_split.ipynb                    <-- APT + Reverse L-PPF trên 3 Datasets
│   ├── apt_3dataset_adapter_ppf_val_split_1.ipynb                   <-- Full Adapter + Uniform PPF trên 3 Datasets
│   ├── apt_3dataset_adapter_lppf_val_split_1.ipynb                  <-- Full Adapter + L-PPF trên 3 Datasets
│   ├── apt_3dataset_adapter_deep6_11_ppf_val_split.ipynb            <-- Deep Adapter (6–11) + Uniform PPF trên 3 Datasets
│   ├── apt_3dataset_adapter_deep6_11_lppf_val_split.ipynb           <-- Deep Adapter (6–11) + L-PPF trên 3 Datasets
│   ├── apt_3dataset_adapter_deep9_11_ppf_val_split.ipynb            <-- Top-3 Adapter (9–11) + Uniform PPF trên 3 Datasets
│   └── apt_3dataset_adapter_deep9_11_lppf_val_split.ipynb           <-- Top-3 Adapter (9–11) + L-PPF trên 3 Datasets
│
└── Result/                                              <-- [Dữ liệu thực nghiệm & Trực quan hóa]
    └── APT_CUB200_Experiments_Results.xlsx                         <-- Bảng số liệu Excel trên CUB200 (Mean ± Std trên 3 Seeds)

```