# ==============================================================================
# 5. CHẠY: mỗi (dataset, seed) -> thư mục đặc trưng Swin (cùng định dạng dump của APT) -> phân tích
#    Có dump KAC của phiên APT (Add Input) : Swin proto/fuse/hier + ViT proto/fuse/hier + KAC + oracle, cùng ảnh, cùng nhãn
#    Không có dump                          : chỉ Swin proto (không cần KAC)
# ==============================================================================
import shutil, time
import pandas as pd

def find_dumps(ds, seed):
    for d in sorted(glob.glob(f"{INPUT_ROOT}/**/runs/kac/{ds}/{PHASE}/dumps/seed{seed}", recursive=True)):
        T = len(glob.glob(f"{d}/train_frozen_*.pt"))
        if T and all(os.path.exists(f"{d}/test_{t}_{j}.pt") for t in range(T) for j in range(t + 1)) \
             and all(os.path.exists(f"{d}/test_frozen_{j}.pt") for j in range(T)):
            return d, T
    return None, None

def write_swin_dir(ds, seed, dump, name):
    tr, tr_y, ev, ev_y = seed_split(ds, seed)
    msg = check_against_json(ds, seed, ev, ev_y)
    Fd = features(ds, name)
    b = np.cumsum([0] + DS_SPEC[ds]["sizes"]); T = len(b) - 1
    d = f"{OUT}/work/{ds}/{PHASE}/{BACKBONES[name]}/seed{seed}"; shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    for t in range(T):
        for name, items, ys in (("train", tr, tr_y), ("test", ev, ev_y)):
            sel = [n for n, y in enumerate(ys) if b[t] <= y < b[t + 1]]          # cùng thứ tự với repo APT (lọc theo task, giữ thứ tự json)
            f = torch.stack([Fd[items[n][0]][items[n][1]] for n in sel])
            torch.save({"f": f, "y": torch.tensor([ys[n] for n in sel])}, f"{d}/{name}_frozen_{t}.pt")
    if dump:
        for t in range(T):
            for j in range(t + 1): os.symlink(f"{dump}/test_{t}_{j}.pt", f"{d}/test_{t}_{j}.pt")
    return d, T, msg

def analyse_proto_only(d, T, seed, ds):
    """Spherical-Swin thuần khi không có dump KAC (cùng cách tính với analyse_seed_src: cùng tâm, cùng κ)."""
    ld = lambda n: torch.load(os.path.join(d, n))
    b = np.cumsum([0] + DS_SPEC[ds]["sizes"]); bounds = [(int(b[t]), int(b[t + 1])) for t in range(T)]; C = int(b[-1])
    tr = [ld(f"train_frozen_{t}.pt") for t in range(T)]
    Ztr = torch.cat([embed(x["f"]) for x in tr]); ytr = torch.cat([x["y"] for x in tr]).long()
    g = torch.Generator().manual_seed(seed)
    cen, has = _build_cen(Ztr, ytr, C, g); inv_med = _inv_med(Ztr)
    te = [ld(f"test_frozen_{j}.pt") for j in range(T)]
    acc, tid = {}, np.zeros((T, T))
    for t in range(T):
        V = bounds[t][1]
        for j in range(t + 1):
            z, y = embed(te[j]["f"]), te[j]["y"].long(); lo, hi = bounds[j]
            for k in K_GRID:
                cos = torch.einsum("nd,ckd->nck", z, cen[k][:V])
                for kp in KAPPA_GRID:
                    s = torch.logsumexp(kp * inv_med * (cos - 1), dim=2); s[:, ~has[:V]] = -1e4
                    acc.setdefault(("fuse", 1.0, kp, k), np.zeros((T, T)))[j, t] = 100.0 * (s.argmax(1) == y).float().mean().item()
                    if k == 1 and kp == 16:
                        pt = s.argmax(1); tid[j, t] = 100.0 * ((pt >= lo) & (pt < hi)).float().mean().item()
    rows = []
    for key, A in acc.items():
        avg, last, forg = _metrics(A, T)
        rows.append(dict(seed=seed, source="frozen", variant="fuse", w=1.0, kappa=key[2], k=key[3], lam=np.nan, w_in=np.nan,
                         avg=avg, last=last, forgetting=forg))
    return rows, dict(seed=seed, source="frozen", task_id_acc_old_last=float(np.mean([tid[j, T - 1] for j in range(T - 1)])),
                      cos_same_image_old_task_last=float("nan"))

ROWS, DIAG = [], []
def add(r, dg, ds, bb, has):
    global ROWS
    for x in r: x.update(dataset=ds, backbone=bb, has_kac=has)
    dg.update(dataset=ds, backbone=bb, has_kac=has); ROWS += r; DIAG.append(dg)

for ds in DATASETS:
    for seed in SEEDS:
        t0 = time.time()
        dump, Td = find_dumps(ds, seed)
        print(f"== {ds} | seed {seed} | dump KAC: {dump or 'KHÔNG CÓ -> chỉ Swin proto'}", flush=True)
        if dump:                                                              # ViT đóng băng: đọc thẳng từ dump
            r, dg = analyse_seed_src(dump, Td, seed, "frozen"); add(r, dg, ds, "ViT", True)
        for name, bb in BACKBONES.items():
            d, T, msg = write_swin_dir(ds, seed, dump, name)
            if dump:
                assert T == Td, f"{ds} seed {seed}: dump có {Td} task, split có {T}"
                r, dg = analyse_seed_src(d, T, seed, "frozen")
            else:
                r, dg = analyse_proto_only(d, T, seed, ds)
            add(r, dg, ds, bb, bool(dump))
        print(f"   split: {msg} | xong sau {time.time() - t0:.0f}s", flush=True)
pd.DataFrame(ROWS).to_csv(f"{OUT}/rows_all_{PHASE}.csv", index=False)
