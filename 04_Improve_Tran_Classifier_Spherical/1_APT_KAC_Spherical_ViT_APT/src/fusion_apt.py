"""Spherical branch on three feature sources, same KAC logits, same grids (needs fusion_online.py code in the same namespace).
  frozen      : ViT without prompt. Centres = train images of each task (once). Test features = frozen features.
  apt_stale   : APT features (CLS after the prompts). Centre of a class = APT features of its train images taken right after
                its own task and never refreshed; test features = APT features of the CURRENT model (what a real continual run has).
  apt_refresh : at every step, centres rebuilt from the current model's features of ALL seen train images.
                A real continual run does not keep old train images -> reference only (separates drift from feature quality).
Dumps: train_frozen_{t}.pt, train_apt_{t}_{j}.pt, test_frozen_{j}.pt, test_apt_{t}_{j}.pt, test_{t}_{j}.pt (see apt_patch.py)."""

SOURCES = ["frozen", "apt_stale", "apt_refresh"]


def _build_cen(Z, y, C, g):
    has = torch.zeros(C, dtype=torch.bool); has[y.unique()] = True
    cen = {k: torch.zeros(C, k, Z.shape[1]) for k in K_GRID}
    for c in has.nonzero().flatten().tolist():
        Zc = Z[y == c]
        for k in K_GRID:
            Cc = sph_kmeans(Zc, k, g); cen[k][c] = Cc[torch.arange(k) % Cc.shape[0]]
    return cen, has


def _inv_med(Z):
    z0 = Z[:512]; G = 1 - z0 @ z0.T
    return 1.0 / G[torch.triu(torch.ones(len(z0), len(z0), dtype=torch.bool), 1)].median().item()


def sources_available(d, T):
    ex = lambda n: os.path.exists(os.path.join(d, n))
    out = ["frozen"]
    if all(ex(f"train_apt_{t}_{t}.pt") for t in range(T)) and all(ex(f"test_apt_{t}_{j}.pt") for t in range(T) for j in range(t + 1)):
        out.append("apt_stale")
        if all(ex(f"train_apt_{t}_{j}.pt") for t in range(T) for j in range(t + 1)): out.append("apt_refresh")
    return out


def analyse_seed_src(d, T, seed, source):
    ld = lambda n: torch.load(os.path.join(d, n))
    Vs = [ld(f"test_{t}_{t}.pt")["k"].shape[1] for t in range(T)]
    bounds_all = [((Vs[t - 1] if t else 0), Vs[t]) for t in range(T)]
    C = Vs[-1]
    te_y = [ld(f"test_frozen_{j}.pt")["y"].long() for j in range(T)]
    g = torch.Generator().manual_seed(seed)
    if source == "frozen":
        tr = [ld(f"train_frozen_{t}.pt") for t in range(T)]
        zte_cache = [embed(ld(f"test_frozen_{j}.pt")["f"]) for j in range(T)]
        zte = lambda t, j: zte_cache[j]
    else:
        tr = [ld(f"train_apt_{t}_{t}.pt") for t in range(T)]
        zte = lambda t, j: embed(ld(f"test_apt_{t}_{j}.pt")["f"])
    for t, x in enumerate(tr):
        lo, hi = bounds_all[t]; assert x["y"].min() >= lo and x["y"].max() < hi, f"task {t}: nhãn train ngoài [{lo},{hi})"
    if source != "apt_refresh":                                   # centres built once, from each task's own train images
        Ztr = torch.cat([embed(x["f"]) for x in tr]); ytr = torch.cat([x["y"] for x in tr]).long()
        cen_s, has_s = _build_cen(Ztr, ytr, C, g); inv_s = _inv_med(Ztr)
        if source == "apt_stale":
            fz = [ld(f"train_frozen_{t}.pt")["y"] for t in range(T)]
            assert all(torch.equal(a["y"], b) for a, b in zip(tr, fz)), "train_apt và train_frozen lệch thứ tự"
    acc, tid = {}, np.zeros((T, T)); drift = np.full((T, T), np.nan)
    def put(key, j, t, pred, y):
        acc.setdefault(key, np.zeros((T, T)))[j, t] = 100.0 * (pred == y).float().mean().item()
    for t in range(T):
        bt = bounds_all[:t + 1]; V = bt[-1][1]
        if source == "apt_refresh":                               # re-embed everything seen so far with the current model
            parts = [ld(f"train_apt_{t}_{j}.pt") for j in range(t + 1)]
            Zt = torch.cat([embed(x["f"]) for x in parts]); yt = torch.cat([x["y"] for x in parts]).long()
            cen, has = _build_cen(Zt, yt, V, g); inv_med = _inv_med(embed(parts[0]["f"]))
        else:
            cen, has, inv_med = cen_s, has_s, inv_s
        for j in range(t + 1):
            v = ld(f"test_{t}_{j}.pt"); kac, y = v["k"].float(), v["y"].long()
            assert kac.shape[1] == V and torch.equal(y, te_y[j]), f"seed {seed} step {t} task {j}: nhãn/đặc trưng không khớp"
            z = zte(t, j); lo, hi = bounds_all[j]
            if source == "apt_stale" and j < t:                   # how far did the same test images move since task j?
                drift[j, t] = (z * zte(j, j)).sum(1).mean().item()
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
        r = dict(seed=seed, source=source, variant=key[0], w=np.nan, kappa=np.nan, k=np.nan, lam=np.nan, w_in=np.nan, avg=avg, last=last, forgetting=forg)
        if key[0] == "fuse": r.update(w=key[1], kappa=key[2], k=key[3])
        if key[0] == "hier": r.update(lam=key[1], w_in=key[2], kappa=key[3], k=key[4])
        rows.append(r)
    dg = dict(seed=seed, source=source,
              task_id_acc_old_last=float(np.mean([tid[j, T - 1] for j in range(T - 1)])) if T > 1 else float("nan"),
              cos_same_image_old_task_last=float(np.nanmean([drift[j, T - 1] for j in range(T - 1)])) if (source == "apt_stale" and T > 1) else float("nan"))
    return rows, dg


def analyse_all(run_dir, sources=SOURCES):
    rows, diag = [], []
    for seed, (d, T) in sorted(seeds_complete(run_dir).items()):
        have = sources_available(d, T)
        for s in sources:
            if s not in have:
                print(f"  seed {seed}: thiếu dump cho nguồn {s} (cần APT_DUMP_APT={'2' if s == 'apt_refresh' else '1'}) - bỏ qua"); continue
            print(f"  seed {seed}: {T} task, nguồn {s}", flush=True)
            r, dg = analyse_seed_src(d, T, seed, s); rows += r; diag.append(dg)
    return rows, diag
