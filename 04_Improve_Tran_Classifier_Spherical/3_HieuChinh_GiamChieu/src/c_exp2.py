# ==============================================================================
# 5. THỬ NGHIỆM 2: GIẢM CHIỀU ĐẶC TRƯNG SPHERICAL (ViT đóng băng) 768 -> d
#    z0 = LayerNorm(f); p = (z0 - μ) W  (W: 768 x d); z = p / ||p||  -> spherical k-means, vMF như cũ
#    pca : μ, W học từ ảnh train của TASK 1 rồi giữ nguyên (không nhìn dữ liệu task sau)
#    rp  : μ = 0, W = ma trận trực giao ngẫu nhiên (seed cố định), không cần dữ liệu
#    "full" (d = 768, không chiếu) chính là Spherical-ViT đã có -> chạy lại để so cặp.
# ==============================================================================
_embed_orig = embed
def ln(f): return F.layer_norm(f.float(), (f.shape[1],))

def proj_matrix(kind, d, Z0_task1, seed):
    D = Z0_task1.shape[1]
    if kind == "pca":
        mu = Z0_task1.mean(0)
        U, S, Vh = torch.linalg.svd(Z0_task1 - mu, full_matrices=False)
        return mu, Vh[:d].T.contiguous(), float((S[:d] ** 2).sum() / (S ** 2).sum())
    g = torch.Generator().manual_seed(1000 + seed)
    Q, _ = torch.linalg.qr(torch.randn(D, d, generator=g))
    return torch.zeros(D), Q, float("nan")

ROWS2, EVR = [], []
t_all = time.time()
for (ds, seed), (d, T) in RUNS.items():
    tr = [torch.load(f"{d}/train_frozen_{t}.pt") for t in range(T)]
    te = [torch.load(f"{d}/test_frozen_{j}.pt") for j in range(T)]
    kac_files = [f"test_{t}_{j}.pt" for t in range(T) for j in range(t + 1)]
    # full = Spherical-ViT gốc
    globals()["embed"] = _embed_orig
    r, _ = analyse_seed_src(d, T, seed, "frozen")
    for x in r: x.update(dataset=ds, proj="full", dim=768)
    ROWS2 += [x for x in r if x["variant"] in ("fuse", "hier")]
    for kind in PROJ_TYPES:
        for dim in PROJ_DIMS:
            mu, W, evr = proj_matrix(kind, dim, ln(tr[0]["f"]), seed)
            EVR.append(dict(dataset=ds, seed=seed, proj=kind, dim=dim, explained_var_task1=evr))
            w = f"{OUT}/work/proj/{ds}/seed{seed}/{kind}{dim}"
            link_dir(w, d, kac_files)
            for t in range(T):
                torch.save({"f": (ln(tr[t]["f"]) - mu) @ W, "y": tr[t]["y"]}, f"{w}/train_frozen_{t}.pt")
                torch.save({"f": (ln(te[t]["f"]) - mu) @ W, "y": te[t]["y"]}, f"{w}/test_frozen_{t}.pt")
            globals()["embed"] = lambda f: F.normalize(f.float(), dim=1)        # đã LayerNorm trước khi chiếu
            r, _ = analyse_seed_src(w, T, seed, "frozen")
            globals()["embed"] = _embed_orig
            for x in r: x.update(dataset=ds, proj=kind, dim=dim)
            ROWS2 += [x for x in r if x["variant"] in ("fuse", "hier")]
    print(f"{ds:13} seed {seed}: xong ({time.time() - t_all:.0f}s)", flush=True)
globals()["embed"] = _embed_orig
pd.DataFrame(ROWS2).to_csv(f"{OUT}/exp2_proj_rows.csv", index=False)
