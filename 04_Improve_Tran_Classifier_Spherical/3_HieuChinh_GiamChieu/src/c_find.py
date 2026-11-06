# ==============================================================================
# 3. TÌM DUMP (output các notebook APT đã Add Input)
# ==============================================================================
import shutil, time, glob
import pandas as pd

def find_dumps(ds, seed):
    for d in sorted(glob.glob(f"{INPUT_ROOT}/**/runs/kac/{ds}/{PHASE}/dumps/seed{seed}", recursive=True)):
        T = len(glob.glob(f"{d}/train_frozen_*.pt"))
        if T and all(os.path.exists(f"{d}/test_{t}_{j}.pt") for t in range(T) for j in range(t + 1)) \
             and all(os.path.exists(f"{d}/test_frozen_{j}.pt") for j in range(T)):
            return d, T
    return None, None

RUNS = {}
for ds in DATASETS:
    for seed in SEEDS:
        d, T = find_dumps(ds, seed)
        print(f"{ds:13} seed {seed}: {d or 'KHÔNG CÓ DUMP -> bỏ qua'}")
        if d: RUNS[(ds, seed)] = (d, T)
assert RUNS, "Không thấy dump nào: Add Input output các notebook APT (chứa runs/kac/<dataset>/val/dumps/seed*)"

def bounds_of(d, T):
    Vs = [torch.load(f"{d}/test_{t}_{t}.pt")["k"].shape[1] for t in range(T)]
    return [((Vs[t - 1] if t else 0), Vs[t]) for t in range(T)]

def link_dir(dst, src, names):
    shutil.rmtree(dst, ignore_errors=True); os.makedirs(dst)
    for n in names: os.symlink(f"{src}/{n}", f"{dst}/{n}")
