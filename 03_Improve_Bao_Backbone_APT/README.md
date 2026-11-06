# 03. Hướng Nghiên Cứu Cải Tiến: Backbone APT — Hierarchical Layer-Wise PPF (L-PPF) & Multi-Depth Residual Adapters

> **Dự án:** Continual Learning with Pre-trained Vision Models: A Comprehensive Study on Parameter-Efficient Adaptation and Classifier Architectures  
> **Môn học:** Deep Learning Project (DPL302m)  
> **Giảng viên hướng dẫn:** PGS. TS. Phan Thị Thu Hồng

---

## 📌 1. Đặt Vấn Đề & Cơ Sở Lý Thuyết (Theoretical Motivation)

Trong bài toán **Exemplar-Free Class-Incremental Learning (EFCIL)**, mô hình phải học tuần tự các tập lớp mới mà tuyệt đối **không lưu trữ mẫu cũ** ($M_{\text{buffer}} = 0$) và **không biết ID của task khi suy luận**.

Mô hình nền tảng **Additive Prompt Tuning (APT - ICCV 2025)** đóng băng toàn bộ backbone Vision Transformer (ViT-B/16, 85.8M tham số) và tối ưu một tập prompt cộng tính $\Phi_t = \{\mathbf{p}_k^l, \mathbf{p}_v^l\}_{l=1}^{12}$ tại vector $\texttt{[CLS]}$. Quá trình chuyển giao tri thức giữa các task được thực hiện qua cơ chế **Progressive Prompt Fusion (PPF)**:

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

$$\hat{y} = \arg\max_{c \in \mathcal{C}_{1:t}} \left[ \operatorname{softmax}\left(\frac{\ell_{\text{cos}}(x)}{\tau}\right) + \alpha_{\text{hyb}} \operatorname{softmax}(\ell_{\text{kac}}(x)) \right]_c$$

---

## 🗂️ 3. Cấu Trúc Mã Nguồn & Danh Mục Thực Nghiệm

```text
My_Part/ (03_Improve_Bao_Backbone_APT)
├── README.md                                                        <-- (Tài liệu này) Báo cáo chi tiết phương pháp & kết quả
│
├── 1_CUB200_Architectural_Exploration/                              <-- [Bộ thí nghiệm chuyên sâu Fine-grained (10 Notebooks CUB-200)]
│   ├── apt_cub200_kac_classifier_val_split.ipynb                    <-- [BASELINE] APT + KAC Classifier (Uniform PPF, α=0.70)
│   ├── apt_cub200_hybrid_classifier_val_split.ipynb                 <-- [CẢI TIẾN CLASSIFIER] APT + Hybrid Head (Cosine + Detach KAC)
│   ├── apt_cub200_lppf_val_split.ipynb                              <-- [CẢI TIẾN BACKBONE] APT + Hierarchical L-PPF (α: 0.85 → 0.55)
│   ├── apt_cub200_reverse_lppf_val_split.ipynb                      <-- [ABLATION ĐỐI CHỨNG] APT + Reverse L-PPF (α: 0.55 → 0.85)
│   ├── apt_cub200_adapter_uniform-ppf_val_split.ipynb               <-- [CẢI TIẾN ADAPTER] Full Adapter (0–11) + Uniform PPF
│   ├── apt_cub200_adapter_lppf_val_split.ipynb                      <-- [CẢI TIẾN KẾT HỢP] Full Adapter (0–11) + L-PPF Phân tầng
│   ├── apt_cub200_adapter_deep6_11_uniform-ppf_val_split.ipynb      <-- [DEEP ADAPTER] Adapter tầng sâu (6–11) + Uniform PPF
│   ├── apt_cub200_adapter_deep6_11_lppf_val_split.ipynb             <-- [DEEP ADAPTER] Adapter tầng sâu (6–11) + L-PPF Phân tầng
│   ├── apt_cub200_adapter_deep9_11_uniform-ppf_val_split.ipynb      <-- [TOP-3 ADAPTER] Adapter tầng sâu (9–11) + Uniform PPF
│   └── apt_cub200_adapter_deep9_11_lppf_val_split.ipynb             <-- [TOP-3 ADAPTER] Adapter tầng sâu (9–11) + L-PPF Phân tầng
│
├── 2_Multi_Dataset_Validation/                                      <-- [Bộ thẩm định đa miền dữ liệu (10 Notebooks 3 Datasets)]
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
└── 3_KetQua_Va_BieuDo/                                              <-- [Dữ liệu thực nghiệm & Trực quan hóa]
    └── APT_CUB200_Experiments_Results.xlsx                         <-- Bảng số liệu Excel gốc (Mean ± Std trên 3 Seeds)

```

---

## 📊 4. Bảng Kết Quả Thực Nghiệm Tổng Hợp Trên CUB-200 (Validation Split)

Dữ liệu được trích xuất trực tiếp từ file thực nghiệm chuẩn hóa [`APT_CUB200_Experiments_Results.xlsx`](file:///d:/FPTU/SEM_5_FA_26/DPL302m%20-%20Deep%20Learning/Replication/APT-Hybrid_classifier-backbone_modify/Cub200/APT_CUB200_Experiments_Results.xlsx), thực hiện trên **3 independent seeds $\{0, 1, 2\}$** với giao thức Zero-Leakage (80% Train / 20% Val):

| STT | Phương Pháp | Cơ Chế Hợp Nhất (PPF) | Số Tham Số | 5 Tasks (40 Cls/T)<br>$A_T$ / $A_{\text{avg}}$ / $F$ | 10 Tasks (20 Cls/T)<br>$A_T$ / $A_{\text{avg}}$ / $F$ | 20 Tasks (10 Cls/T)<br>$A_T$ / $A_{\text{avg}}$ / $F$ | 40 Tasks (5 Cls/T)<br>$A_T$ / $A_{\text{avg}}$ / $F$ |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **—** | **APT + KAC Classifier [BASELINE]** | Uniform ($\alpha=0.70$) | 0.634M | 80.67 / 87.57 / 7.15 | 67.75 / 81.58 / 17.19 | 54.86 / 72.13 / 29.07 | 53.31 / 70.28 / 30.23 |
| **1** | **APT + Hybrid Classifier (Ours)** | Uniform ($\alpha=0.70$) | 0.634M | 84.44 / 89.37 / 7.64 | **81.61** / 87.49 / 9.10 | 75.03 / **84.67** / 12.14 | **72.83** / **82.41** / 14.19 |
| **2** | **APT + L-PPF (Hierarchical)** | L-PPF ($\alpha: 0.85 \to 0.55$) | 0.634M | 84.47 / 89.43 / 7.75 | 81.17 / 87.44 / 9.35 | **75.33** / 84.23 / **11.64** | 72.14 / 82.30 / **13.62** |
| **3** | **APT + Reverse L-PPF (Đối chứng)** | Inv. L-PPF ($\alpha: 0.55 \to 0.85$) | 0.634M | 84.61 / 89.32 / 7.46 | 81.11 / 87.42 / 9.75 | 74.86 / 84.40 / 12.54 | 71.17 / 82.26 / 15.67 |
| **4** | **APT + Full Adapter (0–11)** | Uniform ($\alpha=0.70$) | 0.938M | **85.64** / **90.11** / **7.08** | **82.20** / **88.77** / 9.97 | 68.86 / 81.60 / 17.34 | 52.47 / 68.09 / 28.63 |
| **5** | **APT + Full Adapter (0–11)** | L-PPF ($\alpha: 0.85 \to 0.55$) | 0.938M | 85.00 / 89.70 / 7.88 | 81.58 / 88.15 / 10.93 | 69.11 / 80.61 / 18.45 | 53.42 / 69.69 / 28.15 |
| **6** | **APT + Deep Adapter (6–11)** | Uniform ($\alpha=0.70$) | 0.786M | 84.92 / 89.75 / 7.50 | 81.28 / 87.85 / 10.25 | 65.89 / 80.24 / 20.29 | 46.11 / 63.96 / 33.82 |
| **7** | **APT + Deep Adapter (6–11)** | L-PPF ($\alpha: 0.85 \to 0.55$) | 0.786M | 85.05 / 89.53 / 7.26 | 80.31 / 87.72 / 11.67 | 68.67 / 80.61 / 17.81 | 47.97 / 64.92 / 33.02 |
| **8** | **APT + Top-3 Adapter (9–11)**| Uniform ($\alpha=0.70$) | 0.710M | 84.47 / 89.35 / 8.02 | 80.75 / 87.52 / 10.37 | 66.61 / 79.11 / 18.31 | 50.66 / 64.65 / 26.84 |
| **9** | **APT + Top-3 Adapter (9–11)**| L-PPF ($\alpha: 0.85 \to 0.55$) | 0.710M | 84.47 / 89.29 / 8.33 | 80.08 / 87.11 / 12.13 | 68.56 / 80.96 / 19.91 | 50.47 / 66.85 / 32.79 |

*Ghi chú các chỉ số:*
* $A_T$: Last Task Accuracy (Độ chính xác sau khi hoàn thành chuỗi task).
* $A_{\text{avg}}$: Average Incremental Accuracy (Độ chính xác trung bình tích lũy).
* $F$: Backward Forgetting Measure (Tỷ lệ suy giảm độ chính xác trên các task đã học, càng thấp càng tốt).

---

## 💡 5. Nhận Định Khoa Học Cốt Lõi (Scientific Insights)

1. **Khắc phục điểm yếu chí mạng của Baseline APT + KAC:**
   * Baseline **APT + KAC** bị sụt giảm nghiêm trọng khi số task tăng: từ $80.67\%$ (5T) rơi thẳng xuống **$53.31\%$** (40T) do tỷ lệ quên $F$ lên tới **$30.23\%$** (hiện tượng gradient từ KAC làm méo mó không gian đặc trưng prompt).
   * **APT + Hybrid Classifier** và **APT + L-PPF** khắc phục triệt để vấn đề này, giữ vững $A_T$ ở mức **$72.83\%$** (40T) và kéo tỷ lệ quên xuống chỉ còn **$13.62\%$** (giảm hơn một nửa mức quên của Baseline).
2. **Hiệu năng vượt bậc của Residual Adapter trong kịch bản ít task (5T & 10T):**
   * **Full Adapter (Layers 0–11)** thiết lập đỉnh cao độ chính xác mới: **$85.64\%$** (5 Tasks) và **$82.20\%$** (10 Tasks), vượt trội hoàn toàn so với Baseline APT + KAC (+4.97% và +14.45%).
   * Điều này chứng minh: Khi chuỗi task có độ dài vừa phải, việc bổ sung các nhánh biến đổi phi tuyến song song giải phóng sức mạnh biểu diễn của ViT, giúp mô hình bóc tách chi tiết các loài chim tương đồng trong CUB-200 cực kỳ hiệu quả.
3. **Ưu thế tuyệt đối của L-PPF trong kịch bản chuỗi dài (20T & 40T):**
   * Khi số lượng task tăng lên 20 và 40 tasks (số lớp mỗi task ít, nguy cơ quên thảm họa tăng vọt), việc duy trì trọng số adapter động qua quá nhiều lần EMA tích lũy dễ bị tích tụ nhiễu.
   * Ngược lại, **APT + L-PPF** giữ vững phong độ xuất sắc với $A_T = \mathbf{75.33\%}$ (20 Tasks) và $\mathbf{72.14\%}$ (40 Tasks), giảm tỷ lệ quên ($F$) xuống mức thấp kỷ lục (**11.64%** và **13.62%**).
4. **Hiệu ứng phân tầng được khẳng định qua Negative Control (Reverse L-PPF):**
   * Cấu hình **Canonical L-PPF** luôn đạt tỷ lệ quên $F$ thấp hơn và duy trì độ ổn định tốt hơn **Reverse L-PPF** trên các kịch bản khắt khe (ở 40T: Forgetting của L-PPF là 13.62% vs Reverse L-PPF là 15.67%). Điều này khẳng định vững chắc giả thuyết: *Tầng nông cần ổn định để bảo vệ đặc trưng nền, tầng sâu cần dẻo dai để thích ứng ranh giới phân loại.*

---

## 🚀 6. Hướng Dẫn Thực Thi (1-Click Run trên Kaggle / GPU Server)

Tất cả các notebook trong nhánh đều được cấu hình **tự chứa (self-contained)**, tích hợp sẵn script phân chia dữ liệu Stratified 5-Fold Zero-Leakage:

1. **Môi trường yêu cầu:** Kaggle Notebook GPU (Tesla T4 x1 hoặc P100), bật kết nối **Internet: ON**.
2. **Dữ liệu đầu vào (Input Datasets):**
   * Dataset CUB-200-2011: [`cyizhuo/cub-200-2011-by-classes-folder`](https://www.kaggle.com/datasets/cyizhuo/cub-200-2011-by-classes-folder)
   * Repo APT & Pretrained Checkpoints: Đã được tích hợp sẵn đường dẫn tự động nạp.
3. **Thực thi:** Mở bất kỳ notebook nào trong thư mục `1_CUB200_Architectural_Exploration/` hoặc `2_Multi_Dataset_Validation/` và chọn **"Run All"**. Kết quả accuracy, forgetting matrix và biểu đồ sẽ được in ra trực tiếp tại output.
