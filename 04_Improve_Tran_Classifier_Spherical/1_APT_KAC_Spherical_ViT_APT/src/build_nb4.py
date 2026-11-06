import json
B = "apt_kac_merge"; N = f"{B}/nb4"
rd = lambda p: open(p).read()
ind = lambda t: "".join(("    " + l if l.strip() else l) for l in t.splitlines(True))
def cfg_src():
    t = rd(f"{N}/c_config.py")
    return t.replace('HEADS    = ["kac"]\n', 'HEADS    = ["kac"]\nRUN_TRAINING = True       # False: chỉ phân tích dump có sẵn (Add Input output phiên trước) - không cần GPU, vài phút\nRUN_SMOKE    = True       # False: bỏ ô chạy thử (~10-20 phút; nên tắt với CIFAR-100)\n')
def repo_src():
    t = rd("cell_find_repo.py")
    t = t.replace('assert torch.cuda.is_available(), ', 'assert torch.cuda.is_available() or not RUN_TRAINING, ')
    t = t.replace('assert r.returncode == 0, "Cần bật Internet', 'assert r.returncode == 0 or not RUN_TRAINING, "Cần bật Internet')
    assert t.count("not RUN_TRAINING") == 2; return t
def data_src():
    t = rd(f"{B}/split_data_cell.py"); assert "\nfor ds in DATASETS:\n" in t
    return t.replace("\nfor ds in DATASETS:\n", "\nfor ds in (DATASETS if RUN_TRAINING else []):\n")
def smoke_src():
    t = rd(f"{N}/c_smoke2.py"); head, body = t.split("SMOKE_DS = DATASETS[0]", 1)
    return head + "if RUN_TRAINING and RUN_SMOKE:\n" + ind("SMOKE_DS = DATASETS[0]" + body)
def run_src():
    t = rd(f"{N}/c_run.py"); head, body = t.split("for ds in DATASETS:", 1)
    return head + "if RUN_TRAINING:\n" + ind("for ds in DATASETS:" + body)

mk = lambda t, s: {"cell_type": t, "metadata": {}, "source": s.splitlines(True), **({"execution_count": None, "outputs": []} if t == "code" else {})}
md = lambda s: mk("markdown", s); code = lambda s: mk("code", s)
cells = [
md("""# Cải tiến KAC bằng Spherical: ViT đóng băng và đặc trưng APT — 4 dataset, quy trình Train / Val / Test của nhóm

Code chính thức của APT + head KAC (head lr 1e-3). Sau mỗi task lưu logit KAC và đặc trưng: **ViT đóng băng** (không prompt) và **APT** (CLS sau prompt). Phân tích offline:

| nguồn | tâm lớp | đặc trưng ảnh đánh giá |
|---|---|---|
| `frozen` | ViT không prompt | ViT không prompt |
| `apt_stale` | APT ngay sau task của lớp, không cập nhật | APT của model hiện tại |
| `apt_refresh` | APT của model hiện tại trên mọi ảnh train đã thấy (cần giữ ảnh cũ → chỉ tham chiếu) | APT của model hiện tại |

Mỗi nguồn có `proto` (Spherical thuần), `fuse` (gộp log-xác suất với KAC), `hier` (Spherical chọn task, KAC chọn lớp trong task); thêm `KAC` và `oracle` (KAC biết task id).

**Dữ liệu và quy trình (theo `check-dataset.ipynb` của nhóm):** CIFAR-100 (10×10), CUB-200 (10×20), Stanford Cars (16 + 9×20), ImageNet-R (10×20). Seed s ∈ {0,1,2}: thứ tự lớp = `RandomState(s).permutation`, Val = Fold[s] của 5-fold phân tầng (`base_seed=1993`).
1. **`PHASE = "val"`**: học trên 80% Train, đánh giá trên 20% Val; chọn `k, κ_mult, w, λ, w_in` theo Last trung bình 3 seed → lưu `best_configs_val.json`. Không đụng Test.
2. **`PHASE = "test"`**: học trên 100% Train, đánh giá trên Test, dùng **nguyên** cấu hình đã chọn ở bước 1.

**Kaggle:** Settings → GPU (T4/P100, 1 GPU) + Internet On. Add Input: dataset chứa repo APT (`RepoGit_APT`) và dataset của nhóm cho dataset đang chạy. Kết quả trong `/kaggle/working/apt_spherical/`.
Ước lượng thời gian mỗi seed (T4, fp32 như repo): CUB ~1 giờ, Cars ~2–3 giờ, ImageNet-R ~4 giờ, CIFAR-100 ~8 giờ. Phiên Kaggle tối đa ~12 giờ → nên chạy 1 dataset (CIFAR: 1 seed) mỗi phiên. Phiên sau: Add Input output của phiên trước, ô phân tích sẽ tự gom dump của mọi seed."""),
md("## 1. Cấu hình"), code(cfg_src()),
md("## 2. Tìm repo + cài thư viện\nCần GPU và Internet (tải `timm` và trọng số ViT-B/16)."), code(repo_src()),
md("## 3. Dữ liệu: chia Train / Val / Test theo quy trình của nhóm\nIn ra 5 lớp đầu của Task 1 cho từng seed: phải trùng với `check-dataset.ipynb` (ví dụ CUB seed 0: `[18, 170, 107, 98, 177]`)."), code(data_src()),
md("## 3b. Patch repo và thư viện phân tích\nPatch có assert: repo khác bản đã đọc thì dừng ngay. Repo đọc split của ô 3 qua biến môi trường `APT_SPLIT_DIR`; `rand_split=False` vì thứ tự task đã nằm trong nhãn."),
code(rd(f"{B}/apt_patch.py") + "\n\n" + rd(f"{B}/split_patch.py") + "\n\nprint(apply(REPO)); print(apply_split_patch(REPO))\nsys.path.insert(0, f\"{REPO}\")"),
code(rd(f"{B}/fusion_online.py")), code(rd(f"{B}/fusion_apt.py")),
md("## 4. Hàm phụ và chạy thử nhanh (~10 phút)\nÔ hàm phụ luôn phải chạy. Ô chạy thử: 2 task × 2 epoch trên `DATASETS[0]`, seed `SEEDS[0]` - kiểm tra train + dump + phân tích chạy thông (có thể bỏ qua)."), code(rd(f"{N}/c_helpers.py")), code(smoke_src()),
md("## 5. Chạy thật"), code(run_src()),
md("## 6. Phân tích và bảng kết quả\n`PHASE=\"val\"`: chọn cấu hình theo Last trung bình các seed trên Val, lưu `best_configs_val.json`. `PHASE=\"test\"`: báo cáo trên Test với đúng cấu hình đó."), code("import os, sys, glob, shutil, json\n" + rd(f"{N}/c_report.py")),
md("## 7. Gói kết quả nhỏ để gửi lại"), code(rd(f"{N}/c_zip.py")),
]
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("apt_kac_spherical_4ds_trainvaltest.ipynb", "w"), ensure_ascii=False, indent=1)
for i, c in enumerate(cells):
    if c["cell_type"] == "code": compile("".join(c["source"]), f"cell{i}", "exec")
print(len(cells), "cells OK")
