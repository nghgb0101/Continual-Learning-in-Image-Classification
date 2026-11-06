def calib_kac(run_dir):
    """KAC + hiệu chỉnh thang logit giữa các task (không train, không dùng nhãn).
    Thống kê tính trên logit của toàn bộ ảnh test đang đánh giá ở bước đó (transductive, nhưng không dùng nhãn)."""
    res = {}
    for seed, (d, T) in sorted(seeds_complete(run_dir).items()):
        ld = lambda n: torch.load(os.path.join(d, n))
        Vs = [ld(f"test_{t}_{t}.pt")["k"].shape[1] for t in range(T)]
        bnd = [((Vs[t - 1] if t else 0), Vs[t]) for t in range(T)]
        acc = {}
        def put(key, j, t, pred, y): acc.setdefault(key, np.zeros((T, T)))[j, t] = 100.0 * (pred == y).float().mean().item()
        for t in range(T):
            ks = [ld(f"test_{t}_{j}.pt") for j in range(t + 1)]
            K = torch.cat([v["k"].float() for v in ks]); b = bnd[:t + 1]
            mu_c = K.mean(0, keepdim=True)
            Kb = K.clone(); Kz = K.clone()
            for lo, hi in b:
                m, s = K[:, lo:hi].mean(), K[:, lo:hi].std() + 1e-6
                Kb[:, lo:hi] = K[:, lo:hi] - m; Kz[:, lo:hi] = (K[:, lo:hi] - m) / s
            variants = {"KAC": K, "trừ TB khối": Kb, "z-score khối": Kz, "trừ TB cột": K - mu_c}
            off = 0
            for j, v in enumerate(ks):
                n = len(v["y"]); y = v["y"].long()
                for name, M in variants.items(): put((name,), j, t, M[off:off + n].argmax(1), y)
                off += n
        for key, A in acc.items():
            res.setdefault(key[0], []).append(fo_metrics(A, T))
    print(f"{'cách':<14}{'Avg':>16}{'Last':>16}{'Forgetting':>16}   (trung bình ± std qua {len(next(iter(res.values())))} seed)")
    for name, L in res.items():
        L = np.array(L); print(f"{name:<14}" + "".join(f"{L[:, i].mean():10.2f} ±{L[:, i].std():4.2f}" for i in range(3)))

fo_metrics = _metrics
calib_kac(f"{DRIVE_DIR}/runs/kac/CUB200/dumps")
