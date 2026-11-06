# ==============================================================================
# 2. DỮ LIỆU + CHIA SEED (y hệt check-dataset.ipynb / notebook APT của nhóm)
# ==============================================================================
import os, glob, json, pickle
import numpy as np
from torchvision.datasets import ImageFolder

DS_SPEC = {
    "CIFAR100":     dict(C=100, sizes=[10] * 10, kind="cifar",
                         roots=[f"{INPUT_ROOT}/datasets/thohintrnhngc/cifar-100-python/cifar-100-python",
                                f"{INPUT_ROOT}/cifar-100-python/cifar-100-python"]),
    "CUB200":       dict(C=200, sizes=[20] * 10, kind="folder",
                         roots=[f"{INPUT_ROOT}/datasets/cyizhuo/cub-200-2011-by-classes-folder",
                                f"{INPUT_ROOT}/cub-200-2011-by-classes-folder"]),
    "StanfordCars": dict(C=196, sizes=[16] + [20] * 9, kind="folder",
                         roots=[f"{INPUT_ROOT}/datasets/jutrera/stanford-car-dataset-by-classes-folder/car_data/car_data",
                                f"{INPUT_ROOT}/stanford-car-dataset-by-classes-folder/car_data/car_data"]),
    "ImageNet_R":   dict(C=200, sizes=[20] * 10, kind="folder",
                         roots=[f"{INPUT_ROOT}/datasets/thohintrnhngc/imagenet-r", f"{INPUT_ROOT}/imagenet-r"]),
}

def find_root(ds):
    sp = DS_SPEC[ds]
    ok = (lambda p: all(os.path.isfile(os.path.join(p, f)) for f in ("train", "test", "meta"))) if sp["kind"] == "cifar" \
         else (lambda p: os.path.isdir(os.path.join(p, "train")) and os.path.isdir(os.path.join(p, "test")))
    for p in sp["roots"]:
        if ok(p): return p
    tail = os.path.basename(sp["roots"][0].rstrip("/"))
    for p in glob.glob(f"{INPUT_ROOT}/**/{tail}", recursive=True):
        if ok(p): return p
    raise FileNotFoundError(f"{ds}: không thấy dữ liệu. Add Input dataset của nhóm ({sp['roots'][0]})")

def setup_seed(train_targets, seed, num_classes, task_sizes, n_splits=5, base_seed=1993):
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

SRC = {}
def load_source(ds):
    """S["tr"], S["te"]: đường dẫn ảnh (folder) hoặc mảng ảnh uint8 (CIFAR); S["tr_id"]: tên ảnh dùng để đối chiếu json."""
    if ds in SRC: return SRC[ds]
    root, sp = find_root(ds), DS_SPEC[ds]
    if sp["kind"] == "cifar":
        S = dict(root=root)
        for s, name in (("tr", "train"), ("te", "test")):
            with open(os.path.join(root, name), "rb") as f: e = pickle.load(f, encoding="latin1")
            S[s] = e["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1); S[s + "_y"] = [int(v) for v in e["fine_labels"]]
            S[s + "_id"] = [f"{name}:{i}" for i in range(len(S[s + "_y"]))]
    else:
        tr, te = ImageFolder(os.path.join(root, "train")), ImageFolder(os.path.join(root, "test"))
        assert tr.classes == te.classes, f"{ds}: tên lớp train/test không khớp"
        S = dict(root=root, tr=[p for p, _ in tr.samples], tr_y=list(tr.targets), te=[p for p, _ in te.samples], te_y=list(te.targets))
        for s in ("tr", "te"): S[s + "_id"] = [os.path.relpath(p, root) for p in S[s]]
    assert len(set(S["tr_y"])) == sp["C"], f"{ds}: số lớp {len(set(S['tr_y']))} ≠ {sp['C']}"
    SRC[ds] = S
    print(f"{ds}: {root} | train {len(S['tr_y'])} ảnh, test {len(S['te_y'])} ảnh")
    return S

def seed_split(ds, seed):
    """(train_items, train_labels, eval_items, eval_labels) theo đúng thứ tự file json của nhóm.
    item = (split, chỉ số ảnh), split ∈ {"tr","te"}; nhãn = vị trí lớp trong thứ tự task."""
    sp, S = DS_SPEC[ds], load_source(ds)
    task_classes, tr_idx, va_idx = setup_seed(S["tr_y"], seed, sp["C"], sp["sizes"])
    pos = {c: i for i, c in enumerate(sum(task_classes, []))}
    if PHASE == "val":
        tr = [("tr", int(i)) for i in tr_idx]; ev = [("tr", int(i)) for i in va_idx]
    else:
        tr = [("tr", i) for i in range(len(S["tr_y"]))]; ev = [("te", i) for i in range(len(S["te_y"]))]
    lab = lambda it: pos[S[it[0] + "_y"][it[1]]]
    return tr, [lab(x) for x in tr], ev, [lab(x) for x in ev]

def check_against_json(ds, seed, ev, ev_y):
    """Nếu output phiên APT (thư mục splits) được Add Input: so danh sách ảnh đánh giá với file json đã dùng khi train KAC."""
    hits = glob.glob(f"{INPUT_ROOT}/**/splits/{ds}/{PHASE}/seed{seed}/test.json", recursive=True)
    if not hits: return "không có json để đối chiếu"
    js = json.load(open(hits[0])); S = load_source(ds)
    rel = (lambda p: p) if DS_SPEC[ds]["kind"] == "cifar" else \
          (lambda p: os.path.relpath(p, js["root"]) if p.startswith(js["root"]) else p)
    mine = [S[s + "_id"][i] for s, i in ev]
    assert [rel(p) for p in js["data"]] == mine and js["targets"] == ev_y, f"{ds} seed {seed}: split khác file json của phiên APT"
    return "khớp file json của phiên APT"
