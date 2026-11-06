# Code cải tiến KAC bằng nhánh Spherical (Huỳnh Thị Nhã Trân, DPL302m)

> Code APT gốc (của tác giả paper APT, không đưa vào repo này). Bản đã dùng khi chạy được lưu thành Kaggle dataset: <https://www.kaggle.com/datasets/huynhthinhatran/repogit-apt>. Các notebook đọc repo này từ `/kaggle/input` và tự patch khi chạy.

Thư mục này gom toàn bộ code của phần cải tiến được báo cáo trong `5_BaoCao/Report_Spherical_NhaTran.pdf`.

- Toàn bộ số liệu trong report thuộc giai đoạn **Val** (3 seed 0, 1, 2), chưa chạy Test.
- Số liệu độ chính xác đến từ các notebook ở mục 1, 2, 3; số liệu chi phí đến từ notebook ở mục 6.

---

## Tổng quan: cải tiến nào nằm ở đâu

| Cải tiến (theo report) | Notebook | Bảng trong report |
|---|---|---|
| Baseline APT + KAC | `1_APT_KAC_Spherical_ViT_APT/run_*.ipynb` (mỗi file một dataset) | Bảng 1, 2 (cột Baseline) |
| Spherical trên ViT-B/16 đóng băng, kết hợp với KAC (`fuse`, `hier`) | như trên | Bảng 1, 2 |
| Spherical trên đặc trưng APT (`apt_stale`, `apt_refresh`) | như trên | Bảng 1, 2 |
| Spherical trên Swin-B / Swin-T đóng băng, chi phí backbone | `2_Spherical_Swin/swin_spherical_4ds.ipynb` | Bảng 1, 2, 3 |
| Hiệu chỉnh logit KAC theo khối task | `3_HieuChinh_GiamChieu/exp_calib_proj_4ds.ipynb` (thử nghiệm 1) | Bảng 4a, 4b |
| Giảm chiều đặc trưng Spherical (PCA, chiếu ngẫu nhiên) | như trên (thử nghiệm 2) | Bảng 5 |

Thứ tự chạy: các notebook ở mục 1 trước, vì nó sinh ra dump logit KAC và đặc trưng. Notebook 2 và 3 đọc dump này.

```
Notebook 1 (GPU, train APT + KAC, lưu dump)  ──►  dump logit KAC + đặc trưng ViT/APT
        │                                              │
        │                                              ├──► Notebook 2 (GPU, trích đặc trưng Swin, ghép với logit KAC)
        │                                              └──► Notebook 3 (CPU, chỉ đọc dump: hiệu chỉnh logit, giảm chiều)
        └──► bảng Baseline + Spherical ViT/APT
```

Mỗi notebook ở đây là bản đã chạy trên Kaggle (giữ output làm bằng chứng) và chứa sẵn toàn bộ code trong các ô, nên chỉ cần upload notebook lên Kaggle là chạy được. Thư mục `src/` là các file nguồn tương ứng với từng ô, để đọc và sửa cho dễ. `build*.py` là script ghép các file nguồn thành notebook.

---

## 1. `1_APT_KAC_Spherical_ViT_APT/`: APT + KAC + Spherical (ViT đóng băng, APT)

**Notebook:** các file `run_*.ipynb` (Kaggle, GPU T4, bật Internet). Cả 6 file cùng một code, mỗi phiên chạy một dataset; dataset và seed đặt ở ô cấu hình (`DATASETS`, `SEEDS`). Dump logit và đặc trưng của các phiên này là đầu vào của mục 2 và 3.

| Dataset | Seed | File trong thư mục này | Phiên Kaggle |
|---|---|---|---|
| CIFAR-100 | 0 | (không có, xem ghi chú) | dataset `cifar-seed0-dumps` |
| CIFAR-100 | 1 | `run_CIFAR100_seed1.ipynb` | `spherical-kac-cifar` |
| CIFAR-100 | 2 | `run_CIFAR100_seed2.ipynb` | `fork-of-apt-kac-spherical-cifar` |
| CUB-200 | 0, 1, 2 | `run_CUB200.ipynb` | `spherical-kac-cub` |
| Stanford Cars | 0, 1, 2 | `run_Cars.ipynb` | `spherical-kac-cars` |
| ImageNet-R | 0 | (không có, xem ghi chú) | dataset `imagenet-r-seed0-dumps` |
| ImageNet-R | 1 | `run_IMAGENET-R_seed1.ipynb` | `apt-kac-spherical-image-net` |
| ImageNet-R | 2 | `run_IMAGENET-R_seed2.ipynb` | `fork-of-apt-kac-spherical-image-net` |

Ghi chú: seed 0 của CIFAR-100 và ImageNet-R được chạy ở một phiên trước, output (dump, `output.log`, `args.yaml`) được lưu thành Kaggle dataset; notebook của phiên đó không được giữ lại. `src/build_nb4.py` ghép các file trong `src/` thành một notebook gộp 4 dataset (thêm cờ `RUN_TRAINING`, `RUN_SMOKE`); phần train và phân tích giống các file `run_*.ipynb`.

**Đầu vào:**
- Repo APT (Kaggle dataset ở đầu README, add làm input của Kaggle).
- Dataset tương ứng: CIFAR-100, CUB-200, Stanford Cars, ImageNet-R.

**Việc notebook làm:**
1. Chia dữ liệu theo quy trình của nhóm: stratified 5 fold mỗi lớp (`RandomState(1993)`); seed `s` lấy fold `s` làm Val. Thứ tự lớp là `RandomState(s).permutation`.
2. Patch repo APT:
   - thay head bằng KAC (4 RBF, lr head 1e-3);
   - đọc split của nhóm;
   - sau mỗi task lưu logit KAC, đặc trưng ViT đóng băng (không prompt) và đặc trưng APT (CLS sau prompt).
3. Train APT + KAC theo siêu tham số trong `experiments/*.sh` của repo: batch 64, `prompt_param` 0.01. Cấu hình (lr / epoch / EMA) theo dataset:
   - CIFAR: 0.004 / 30 / 0.7
   - CUB: 0.02 / 25 / 0.7
   - Cars: 0.005 / 50 / 0.7
   - ImageNet-R: 0.003 / 30 / 0.8
4. Phân tích offline trên dump:
   - KAC gốc (Baseline);
   - Oracle (biết task id, chỉ dùng làm cận trên);
   - Spherical (`fuse`, `hier`) trên 3 nguồn đặc trưng: `frozen`, `apt_stale`, `apt_refresh`.
5. `PHASE="val"`: với mỗi tổ hợp, chọn cấu hình có Last trung bình 3 seed cao nhất, lưu vào `best_configs_val.json`. `PHASE="test"` dùng lại đúng cấu hình đó; bước này chưa chạy.

**File nguồn (`src/`):**

| File | Ô trong notebook | Nội dung |
|---|---|---|
| `cells/c_config.py` | 2 | Cấu hình: dataset, phase, seed, siêu tham số APT, KAC |
| `cells/cell_find_repo.py` | 4 | Tìm repo APT trong `/kaggle/input`, cài thư viện |
| `split_data_cell.py` | 6 | Chia Train / Val / Test theo quy trình của nhóm, in 5 lớp đầu của Task 1 để đối chiếu |
| `apt_patch.py` + `split_patch.py` | 8 | Patch repo APT: head KAC, dump logit và đặc trưng sau mỗi task, đọc split của nhóm. Có assert: nếu repo khác bản đã đọc thì dừng |
| `fusion_online.py` | 9 | **Lõi Spherical:** `embed` (LayerNorm → L2), `sph_kmeans` (spherical k-means), tính κ, tính Avg / Last / Forgetting |
| `fusion_apt.py` | 10 | **Ghép Spherical với KAC** trên 3 nguồn đặc trưng: `fuse` (công thức 3 trong report), `hier` (công thức 4), Oracle, đo drift của đặc trưng APT |
| `cells/c_helpers.py`, `cells/c_smoke2.py` | 12, 13 | Hàm phụ để chạy lệnh repo, và chạy thử nhanh 2 task × 2 epoch |
| `cells/c_run.py` | 15 | Chạy thật theo dataset × seed |
| `cells/c_report.py` | 17 | Gom kết quả, chọn cấu hình theo Val, xuất CSV |
| `cells/c_zip.py` | 19 | Nén kết quả nhỏ để tải về |
| `diag_kac.py` | (dùng riêng) | Chẩn đoán KAC: độ chính xác từng task, độ chính xác khi biết task id, ảnh bị đoán vào task nào. Đây là bước cho thấy lỗi nằm ở chọn task |
| `calib_kac.py` | (dùng riêng) | Bản đầu của phần hiệu chỉnh logit. Bản dùng cho report nằm ở mục 3 |

**Lưới siêu tham số của Spherical:**
- `k ∈ {1, 3, 5}` (số tâm mỗi lớp)
- `κ_mult ∈ {8, 16, 32, 64}`
- `w ∈ {0.25, 0.5, 0.75, 1.0}`. Giá trị `w = 1` là prototype thuần, tức chỉ dùng Spherical.
- `λ ∈ {0.5, 1, 2}`, `w_in ∈ {0, 0.5}`

Spherical không có tham số nào học bằng gradient.

**Ghi chú về các nguồn đặc trưng:**
- `apt_stale`: tâm của một lớp được tính ngay sau task chứa lớp đó, sau đó không cập nhật nữa. Cấu hình này hợp lệ với bài toán.
- `apt_refresh`: ở mỗi bước, tính lại tâm bằng mô hình hiện tại trên ảnh train cũ. Cách này cần giữ ảnh cũ nên **chỉ để tham chiếu**, không hợp lệ với bài toán không lưu mẫu. Muốn có dump cho nguồn này thì đặt `APT_DUMP_APT="2"`.

---

## 2. `2_Spherical_Swin/`: Spherical trên Swin-B / Swin-T đóng băng

**Notebook:** `swin_spherical_4ds.ipynb` (Kaggle, GPU, bật Internet)

**Đầu vào:**
- Dump của notebook 1, dùng logit KAC (nhánh KAC giữ nguyên).
- Dataset gốc, để trích đặc trưng Swin.

**Việc notebook làm:**
- Thay đặc trưng ViT đóng băng bằng đặc trưng Swin đóng băng trong nhánh Spherical. Mọi thứ khác giữ nguyên: split, k-means, κ, lưới siêu tham số.
- Hai backbone:
  - Swin-B: `swin_base_patch4_window7_224.ms_in22k_ft_in1k`, 1024 chiều.
  - Swin-T: `swin_tiny_patch4_window7_224.ms_in1k`, 768 chiều.

| File (`src/`) | Nội dung |
|---|---|
| `c_config.py` | Cấu hình: backbone, dataset, seed |
| `c_data.py` | Chia seed theo quy trình của nhóm. Phải khớp với notebook 1 để ghép đúng logit KAC |
| `c_feat.py` | Trích đặc trưng Swin; đo tham số, GFLOPs (`FlopCounterMode`) và số ảnh/giây (T4, fp16). Đây là nguồn của Bảng 3 |
| `c_run.py` | Ghép đặc trưng Swin với logit KAC, chạy `fuse` / `hier` (gọi lại `fusion_online.py` và `fusion_apt.py`, đã nhúng sẵn trong notebook); lưu mọi cấu hình vào `rows_all_val.csv` |
| `c_report.py` | Xuất `swin_vs_vit_best_val.csv` (cấu hình tốt nhất), `task_id_acc_val.csv` (độ chính xác chọn task), `backbone_cost.csv` |
| `build.py` | Ghép thành notebook |

---

## 3. `3_HieuChinh_GiamChieu/`: hiệu chỉnh logit KAC và giảm chiều Spherical

**Notebook:** `exp_calib_proj_4ds.ipynb`. Chạy được trên CPU, không cần Internet. Notebook này **không train gì cả**, chỉ đọc dump của notebook 1.

| File (`src/`) | Nội dung |
|---|---|
| `c_find.py` | Tìm dump của 4 dataset × 3 seed |
| `c_exp1.py` | **Thử nghiệm 1, hiệu chỉnh logit KAC theo khối task** (`calibrate`). Có 5 cách: `raw`; `center` (trừ trung bình khối); `center_half` (trừ 0.5 × trung bình khối); `zscore` ((logit − TB khối) / ĐLC khối); `block_softmax` (log-softmax trong từng khối). Chạy cho KAC đứng riêng (Bảng 4a), và KAC đã hiệu chỉnh rồi mới `fuse` với Spherical (Bảng 4b) |
| `c_exp2.py` | **Thử nghiệm 2, giảm chiều**: `z = L2((LN(f) − μ) W)`, với `d ∈ {256, 128, 64}`. PCA: μ và W lấy từ ảnh train của task 1. Chiếu ngẫu nhiên: μ = 0, W là ma trận trực giao ngẫu nhiên (QR của ma trận Gaussian, seed `1000 + seed`). Cuối cùng chạy lại `full` (768 chiều) để so sánh từng cặp |
| `c_report.py` | Xuất `exp_summary_val.csv`, `pca_explained_var.csv` |
| `c_config.py`, `c_zip.py`, `build.py` | Cấu hình, nén kết quả, ghép notebook |

---

## 4. `4_KetQua/`: số liệu đã dùng trong report (Val, 3 seed)

| File | Nội dung |
|---|---|
| `ket_qua_chinh_val.csv` | Cấu hình tốt nhất cho từng (dataset, nguồn, biến thể): Avg, Last, Forgetting (TB ± ĐLC) và cấu hình đã chọn. Nguồn của Bảng 1, 2 |
| `last_theo_seed_val.csv` | Last của từng seed (Baseline, ViT đóng băng, Swin). Nguồn của Phụ lục B |
| `hieu_chinh_giam_chieu_tom_tat.csv`, `exp1_calib_kac_rows.csv`, `pca_explained_var.csv` | Nguồn của Bảng 4a, 4b, 5 |
| `backbone_cost.csv` | Nguồn của Bảng 3 |
| `spherical_4ds_vit_swin_apt_val.csv`, `tong_hop_spherical_val.csv` | Bảng gộp thô từ output Kaggle (ViT, APT, Swin), dùng để tạo `ket_qua_chinh_val.csv` |

## 5. `5_BaoCao/`

Gồm report PDF, file `.tex` (biên dịch bằng XeLaTeX, cần `hinh1.png` cùng thư mục) và `tables.py`. Script `tables.py` sinh các bảng LaTeX trực tiếp từ CSV, đồng thời kiểm tra trung bình theo seed có khớp với bảng chính không.

## 6. `6_DoChiPhi/`: đo chi phí tính toán

**Notebook:** `cost_profile_4ds.ipynb` (Kaggle, GPU T4). Không train đầy đủ: mỗi dataset train Task 1 vài epoch để đo tốc độ rồi ngoại suy. Đo thời gian train, VRAM, số tham số, thời gian xây tâm Spherical, GFLOPs và ms/ảnh khi suy luận của từng cấu hình. Kết quả nằm ở `ket_qua/` (`cost_train.csv`, `cost_spherical_build.csv`, `cost_inference.csv`). File nguồn ở `src/`.
