"""Offline analysis of the dumps written by the patched APT repo (apt_patch.py).
Per seed dir:  train_frozen_{t}.pt {f,y}   test_frozen_{j}.pt {f,y}   test_{t}_{j}.pt {k,y}  (k = logits over the t+1 seen tasks).
Labels are already in task order (class c of task j lies in [lo_j, hi_j)), frozen/KAC rows of a test set are in the same order."""
import os, glob, math
import numpy as np
import torch
import torch.nn.functional as F

W_GRID = [0.25, 0.5, 0.75, 1.0]
KAPPA_GRID = [8, 16, 32, 64]
K_GRID = [1, 3, 5]
LAM_GRID = [0.5, 1.0, 2.0]
WIN_GRID = [0.0, 0.5]


def embed(f):
    return F.normalize(F.layer_norm(f.float(), (f.shape[1],)), dim=1)


def sph_kmeans(Z, k, g, iters=20):
    k = min(k, Z.shape[0])
    C = Z[torch.randperm(Z.shape[0], generator=g)[:k]].clone()
    for _ in range(iters):
        a = (Z @ C.T).argmax(1)
        newC = torch.zeros_like(C).index_add_(0, a, Z)
        empty = newC.norm(dim=1) < 1e-8
        if empty.any():
            newC[empty] = Z[torch.randint(0, Z.shape[0], (int(empty.sum()),), generator=g)]
        C = F.normalize(newC, dim=1)
    return C


def seeds_complete(run_dir):
    res = {}
    for d in sorted(glob.glob(os.path.join(run_dir, "seed*"))):
        T = len(glob.glob(os.path.join(d, "train_frozen_*.pt")))
        if T and all(os.path.exists(os.path.join(d, f"test_{t}_{j}.pt")) for t in range(T) for j in range(t + 1)) \
           and all(os.path.exists(os.path.join(d, f"test_frozen_{j}.pt")) for j in range(T)):
            res[int(os.path.basename(d)[4:])] = (d, T)
    return res


def block_lsm(x, bounds):
    out = torch.empty_like(x)
    for lo, hi in bounds:
        out[:, lo:hi] = F.log_softmax(x[:, lo:hi], 1)
    return out


def block_lse(ls, bounds):
    out = torch.empty_like(ls)
    for lo, hi in bounds:
        out[:, lo:hi] = torch.logsumexp(ls[:, lo:hi], 1, keepdim=True)
    return out


def _metrics(A, T):
    at = [A[:t + 1, t].mean() for t in range(T)]
    forg = np.mean([A[j, j] - A[j, T - 1] for j in range(T - 1)]) if T > 1 else 0.0   # = repo's calc_general_forgetting
    return float(np.mean(at)), float(at[-1]), float(forg)


def analyse_seed(d, T, seed):
    ld = lambda n: torch.load(os.path.join(d, n))
    tr = [ld(f"train_frozen_{t}.pt") for t in range(T)]
    te = [ld(f"test_frozen_{j}.pt") for j in range(T)]
    # ranh giới task lấy từ độ rộng logit (repo CUB có nhãn 1..200 nên 1 lớp rỗng, task chứa nó có 19 lớp nhưng vẫn chiếm 20 cột)
    Vs = [ld(f"test_{t}_{t}.pt")["k"].shape[1] for t in range(T)]
    bounds_all = [((Vs[t - 1] if t else 0), Vs[t]) for t in range(T)]
    C = Vs[-1]
    Ztr = torch.cat([embed(x["f"]) for x in tr]); ytr = torch.cat([x["y"] for x in tr]).long()
    for t, x in enumerate(tr):
        lo, hi = bounds_all[t]; assert x["y"].min() >= lo and x["y"].max() < hi, f"task {t}: nhãn train ngoài [{lo},{hi})"
    has = torch.zeros(C, dtype=torch.bool); has[ytr.unique()] = True
    print(f"    {int((~has).sum())} lớp không có ảnh train (bỏ qua):", (~has).nonzero().flatten().tolist())
    z0 = Ztr[:512]
    G = 1 - z0 @ z0.T
    inv_med = 1.0 / G[torch.triu(torch.ones(len(z0), len(z0), dtype=torch.bool), 1)].median().item()
    g = torch.Generator().manual_seed(seed)
    cen = {k: torch.zeros(C, k, Ztr.shape[1]) for k in K_GRID}
    for c in range(C):
        if not has[c]: continue
        Zc = Ztr[ytr == c]
        for k in K_GRID:
            Cc = sph_kmeans(Zc, k, g)
            cen[k][c] = Cc[torch.arange(k) % Cc.shape[0]]
    acc, tid = {}, np.zeros((T, T))
    def put(key, j, t, pred, y):
        acc.setdefault(key, np.zeros((T, T)))[j, t] = 100.0 * (pred == y).float().mean().item()
    for t in range(T):
        bt = bounds_all[:t + 1]; V = bt[-1][1]
        for j in range(t + 1):
            v = ld(f"test_{t}_{j}.pt"); kac, y = v["k"].float(), v["y"].long()
            assert kac.shape[1] == V and torch.equal(y, te[j]["y"].long()), f"seed {seed} step {t} task {j}: nhãn/đặc trưng không khớp"
            z = embed(te[j]["f"]); lo, hi = bounds_all[j]
            lk = F.log_softmax(kac, 1); lin_k = block_lsm(kac, bt)
            put(("KAC",), j, t, kac.argmax(1), y)
            blk = torch.full_like(kac, -1e9); blk[:, lo:hi] = kac[:, lo:hi]
            put(("oracle",), j, t, blk.argmax(1), y)
            for k in K_GRID:
                cos = torch.einsum("nd,ckd->nck", z, cen[k][:V])
                for kp in KAPPA_GRID:
                    s = torch.logsumexp(kp * inv_med * (cos - 1), dim=2)
                    s[:, ~has[:V]] = -1e4
                    ls = F.log_softmax(s, 1)
                    for w in W_GRID:
                        put(("fuse", w, kp, k), j, t, ((1 - w) * lk + w * ls).argmax(1), y)
                    lt = block_lse(ls, bt); lin_s = block_lsm(s, bt)
                    for lam in LAM_GRID:
                        for w_in in WIN_GRID:
                            put(("hier", lam, w_in, kp, k), j, t, (lam * lt + (1 - w_in) * lin_k + w_in * lin_s).argmax(1), y)
                    if k == 1 and kp == 16:
                        pt = s.argmax(1); tid[j, t] = 100.0 * ((pt >= lo) & (pt < hi)).float().mean().item()
    rows = []
    for key, A in acc.items():
        avg, last, forg = _metrics(A, T)
        r = dict(seed=seed, variant=key[0], w=np.nan, kappa=np.nan, k=np.nan, lam=np.nan, w_in=np.nan, avg=avg, last=last, forgetting=forg)
        if key[0] == "fuse": r.update(w=key[1], kappa=key[2], k=key[3])
        if key[0] == "hier": r.update(lam=key[1], w_in=key[2], kappa=key[3], k=key[4])
        rows.append(r)
    return rows, dict(seed=seed, task_id_acc_old_last=float(np.mean([tid[j, T - 1] for j in range(T - 1)])) if T > 1 else float("nan"))


def analyse(run_dir):
    rows, diag = [], []
    for seed, (d, T) in sorted(seeds_complete(run_dir).items()):
        print(f"  seed {seed}: {T} task", flush=True)
        r, dg = analyse_seed(d, T, seed); rows += r; diag.append(dg)
    return rows, diag
