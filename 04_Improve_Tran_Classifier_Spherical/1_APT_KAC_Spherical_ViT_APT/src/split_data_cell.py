# ==============================================================================
# DỮ LIỆU: 4 dataset + chia Train/Val/Test theo quy trình chung của nhóm (check-dataset.ipynb)
#   - seed s ∈ {0,1,2}: thứ tự lớp = np.random.RandomState(s).permutation(C), cắt theo TASK_SIZES
#   - Val = Fold[s] của Stratified 5-Fold trong từng lớp (base_seed = 1993), Train = 4 fold còn lại
#   - PHASE="val" : học trên 80% Train, đánh giá trên 20% Val   (chọn cấu hình)
#   - PHASE="test": học trên 100% Train,  đánh giá trên Test     (số liệu báo cáo)
# Ghi ra {WORK}/splits/<ds>/<phase>/seed<s>/{train,test}.json; repo APT đọc qua biến môi trường APT_SPLIT_DIR.
# Nhãn được đánh lại = vị trí của lớp trong thứ tự task -> task k là một khối nhãn liên tiếp (rand_split=False trong repo).
# ==============================================================================
import os, json, glob, pickle, random, collections
import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder

WORK = DRIVE_DIR
DS_SPEC = {
    "CIFAR100":     dict(C=100, sizes=[10] * 10,       kind="cifar",
                         roots=["/kaggle/input/datasets/thohintrnhngc/cifar-100-python/cifar-100-python",
                                "/kaggle/input/cifar-100-python/cifar-100-python"]),
    "CUB200":       dict(C=200, sizes=[20] * 10,       kind="folder",
                         roots=["/kaggle/input/datasets/cyizhuo/cub-200-2011-by-classes-folder",
                                "/kaggle/input/cub-200-2011-by-classes-folder"]),
    "StanfordCars": dict(C=196, sizes=[16] + [20] * 9, kind="folder",
                         roots=["/kaggle/input/datasets/jutrera/stanford-car-dataset-by-classes-folder/car_data/car_data",
                                "/kaggle/input/stanford-car-dataset-by-classes-folder/car_data/car_data"]),
    "ImageNet_R":   dict(C=200, sizes=[20] * 10,       kind="folder",
                         roots=["/kaggle/input/datasets/thohintrnhngc/imagenet-r",
                                "/kaggle/input/imagenet-r"]),
}

def find_root(ds):
    sp = DS_SPEC[ds]
    ok = (lambda p: all(os.path.isfile(os.path.join(p, f)) for f in ("train", "test", "meta"))) if sp["kind"] == "cifar" \
         else (lambda p: os.path.isdir(os.path.join(p, "train")) and os.path.isdir(os.path.join(p, "test")))
    for p in sp["roots"]:
        if ok(p): return p
    # dự phòng: tìm trong /kaggle/input theo tên thư mục cuối của đường dẫn chuẩn
    tail = os.path.basename(sp["roots"][0].rstrip("/"))
    for p in glob.glob(f"/kaggle/input/**/{tail}", recursive=True):
        if ok(p): return p
    raise FileNotFoundError(f"{ds}: không thấy dữ liệu. Add Input dataset của nhóm (đường dẫn chuẩn: {sp['roots'][0]})")

def load_source(ds):
    """Trả về (train_items, train_targets, test_items, test_targets, root) theo đúng cách đọc của check-dataset."""
    root = find_root(ds)
    if DS_SPEC[ds]["kind"] == "cifar":
        out = []
        for split in ("train", "test"):
            with open(os.path.join(root, split), "rb") as f:
                entry = pickle.load(f, encoding="latin1")
            y = [int(v) for v in entry["fine_labels"]]
            out += [[f"{split}:{i}" for i in range(len(y))], y]
        return out[0], out[1], out[2], out[3], root
    tr, te = ImageFolder(os.path.join(root, "train")), ImageFolder(os.path.join(root, "test"))
    assert tr.classes == te.classes, f"{ds}: tên lớp train/test không khớp"
    return [p for p, _ in tr.samples], list(tr.targets), [p for p, _ in te.samples], list(te.targets), root

def setup_seed(train_targets, seed, num_classes, task_sizes, n_splits=5, base_seed=1993):
    """Y hệt setup_<dataset>_seed trong check-dataset.ipynb (Phương án A)."""
    assert 0 <= seed < n_splits
    class_order = np.random.RandomState(seed).permutation(num_classes).tolist()
    task_classes, cur = [], 0
    for sz in task_sizes:
        task_classes.append(class_order[cur: cur + sz]); cur += sz
    targets = np.asarray(train_targets)
    tr_idx, va_idx = [], []
    rng_split = np.random.RandomState(base_seed)
    for c in range(num_classes):
        perm = rng_split.permutation(np.where(targets == c)[0])
        folds = np.array_split(perm, n_splits)
        va_idx.extend(folds[seed])
        for k in range(n_splits):
            if k != seed: tr_idx.extend(folds[k])
    return task_classes, np.array(tr_idx), np.array(va_idx)

def split_dir(ds, phase, seed): return f"{WORK}/splits/{ds}/{phase}/seed{seed}"

def write_splits(ds, phase, seeds):
    sp = DS_SPEC[ds]
    tr_items, tr_y, te_items, te_y, root = load_source(ds)
    assert len(set(tr_y)) == sp["C"] and sum(sp["sizes"]) == sp["C"], f"{ds}: số lớp {len(set(tr_y))} ≠ {sp['C']}"
    rows, vals = [], {}
    for s in seeds:
        task_classes, tr_idx, va_idx = setup_seed(tr_y, s, sp["C"], sp["sizes"])
        order = sum(task_classes, [])
        pos = {c: i for i, c in enumerate(order)}                       # nhãn mới = vị trí trong thứ tự task
        if phase == "val":
            tr_sel = [(tr_items[i], tr_y[i]) for i in tr_idx]
            ev_sel = [(tr_items[i], tr_y[i]) for i in va_idx]
        else:
            tr_sel = list(zip(tr_items, tr_y))
            ev_sel = list(zip(te_items, te_y))
        d = split_dir(ds, phase, s); os.makedirs(d, exist_ok=True)
        for name, sel in (("train", tr_sel), ("test", ev_sel)):
            js = {"dataset": ds, "phase": phase, "seed": s, "kind": sp["kind"], "root": root, "class_order": order,
                  "task_sizes": sp["sizes"], "data": [x for x, _ in sel], "targets": [pos[y] for _, y in sel]}
            json.dump(js, open(f"{d}/{name}.json", "w"))
        vals[s] = set(va_idx.tolist())
        rows.append(dict(seed=s, train=len(tr_sel), eval=len(ev_sel), task1_5=task_classes[0][:5],
                         task_sizes=[len(t) for t in task_classes]))
    print(f"✅ {ds} ({phase}) | root: {root} | train gốc {len(tr_items)} / test {len(te_items)} | {sp['C']} lớp")
    for r in rows: print("  ", r)
    if phase == "val" and len(vals) > 1:
        ks = sorted(vals); print("   overlap Val giữa các seed:", [len(vals[a] & vals[b]) for a in ks for b in ks if a < b])
    return tr_items, te_items

def check_decode(ds, phase, seed, n=64):
    """Đọc thử n ảnh ngẫu nhiên của cả hai file json (train, test) để chắc chắn đường dẫn/ảnh dùng được."""
    bad = []
    for name in ("train", "test"):
        js = json.load(open(f"{split_dir(ds, phase, seed)}/{name}.json"))
        pick = random.Random(0).sample(range(len(js["data"])), min(n, len(js["data"])))
        if js["kind"] == "cifar": continue                                # pickle: đã đọc ở trên
        for i in pick:
            try:
                with Image.open(js["data"][i]) as im: im.convert("RGB").load()
            except Exception as e: bad.append((js["data"][i], repr(e)[:60]))
    assert not bad, f"{ds}: ảnh lỗi {bad[:3]}"

for ds in DATASETS:
    write_splits(ds, PHASE, SEEDS)
    check_decode(ds, PHASE, SEEDS[0])
print("Dữ liệu sẵn sàng:", f"{WORK}/splits")
