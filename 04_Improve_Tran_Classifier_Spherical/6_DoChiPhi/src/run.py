# ==============================================================================
# 4. ĐO CHI PHÍ CHO TỪNG DATASET
#    (a) Train APT + KAC trên Task 1: thời gian/epoch thực tế (gồm đọc ảnh), tốc độ GPU thuần, VRAM đỉnh, tham số trainable
#    (b) Nhánh Spherical: thời gian trích đặc trưng + spherical k-means cho Task 1, bộ nhớ tâm của mọi lớp
#    (c) Suy luận: GFLOPs/ảnh, ms/ảnh, VRAM đỉnh cho từng cấu hình
# ==============================================================================
VIT = make_vit().to(DEV)
SWIN_M = {}
def get_swin(name):
    if name not in SWIN_M:
        SWIN_M[name] = timm.create_model(name, pretrained=PRETRAINED, num_classes=0).eval().to(DEV)
        for p in SWIN_M[name].parameters(): p.requires_grad_(False)
    return SWIN_M[name]

TRAIN_ROWS, SPH_ROWS, INF_ROWS = [], [], []
for ds in DATASETS:
    print(f"\n{'=' * 70}\n{ds}\n{'=' * 70}", flush=True)
    sp, S = DS_SPEC[ds], load_source(ds)
    task_classes, tr_idx, va_idx = setup_seed(S["tr_y"], SEED, sp["C"], sp["sizes"])
    pos = {c: i for i, c in enumerate(sum(task_classes, []))}
    c1 = set(task_classes[0]); lo, hi = 0, sp["sizes"][0]
    idx1 = [int(i) for i in tr_idx if S["tr_y"][i] in c1]
    if SMOKE: idx1 = idx1[:8]
    items, ys = [S["tr"][i] for i in idx1], [pos[S["tr_y"][i]] for i in idx1]
    n_train_all, n_val_all, T_ = len(tr_idx), len(va_idx), len(sp["sizes"])

    # ---------------- (a) train APT + KAC trên Task 1 ----------------
    model = APTKAC(VIT, sp["C"]).to(DEV)
    n_prompt = model.prompt.prompt_tokens.numel(); n_head = sum(p.numel() for p in model.head.parameters())
    opt = torch.optim.Adam([{"params": [model.prompt.prompt_tokens], "lr": PROMPT_LR[ds]},
                            {"params": list(model.head.parameters()), "lr": HEAD_LR}])
    scaler = torch.amp.GradScaler("cuda", enabled=(DEV == "cuda"))
    def train_step(x, y):
        loss = F.cross_entropy(model(x)[:, lo:hi], y - lo)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    dl = DataLoader(Imgs(items, ys, TRAIN_TF), batch_size=BATCH_TRAIN, shuffle=True, drop_last=len(items) > BATCH_TRAIN,
                    num_workers=WORKERS, pin_memory=(DEV == "cuda"), persistent_workers=WORKERS > 0)
    model.train(); reset_peak(); ep_t, n_seen = [], 0
    for ep in range(PROFILE_EPOCHS):
        sync(); t0 = time.time(); n_seen = 0
        for x, y in dl:
            train_step(x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)); n_seen += len(y)
        sync(); ep_t.append(time.time() - t0)
        print(f"  train Task 1 epoch {ep + 1}/{PROFILE_EPOCHS}: {ep_t[-1]:.1f}s ({n_seen} ảnh)", flush=True)
    vram_train = peak_gb()
    real_ips = n_seen / ep_t[-1]
    xb = torch.randn(BATCH_TRAIN, 3, 224, 224, device=DEV); yb = torch.randint(lo, hi, (BATCH_TRAIN,), device=DEV)
    gpu_ips = BATCH_TRAIN / time_fn(lambda: train_step(xb, yb))
    model.prompt.merge(); model.eval()
    TRAIN_ROWS.append(dict(dataset=ds, GPU=GPU_NAME, task1_images=len(items), epoch_time_s=ep_t[-1],
        train_img_per_s_real=real_ips, train_img_per_s_gpu=gpu_ips, peak_VRAM_train_GB=vram_train,
        trainable_prompt=n_prompt, trainable_KAC_head=n_head, trainable_total=n_prompt + n_head,
        train_images_all_tasks=n_train_all, full_epochs=FULL_EPOCHS[ds],
        est_full_train_h_real=n_train_all * FULL_EPOCHS[ds] / real_ips / 3600,
        est_full_train_h_gpu=n_train_all * FULL_EPOCHS[ds] / gpu_ips / 3600))

    # ---------------- (b) nhánh Spherical: trích đặc trưng Task 1 + k-means ----------------
    dl_feat = DataLoader(Imgs(items, ys, TEST_TF), batch_size=BATCH_INFER, shuffle=False,
                         num_workers=WORKERS, pin_memory=(DEV == "cuda"))
    # APT (stale) cũng cần 1 lượt forward (có prompt) trên ảnh train của task vừa học để lấy tâm; lúc suy luận thì dùng lại đặc trưng sẵn có
    sources = [("ViT-B/16 đóng băng", lambda x: VIT(x)), ("APT (stale)", lambda x: model.features(x))] + \
              [(lab, (lambda x, n=n: get_swin(n)(x))) for n, lab in SWIN.items()]
    CEN = {}
    for lab, fwd in sources:
        sync(); t0 = time.time(); fs = []
        with torch.no_grad():
            for x, _ in dl_feat:
                with torch.autocast(DEV, dtype=torch.float16, enabled=(DEV == "cuda")):
                    fs.append(fwd(x.to(DEV, non_blocking=True)).float())
        sync(); t_feat = time.time() - t0
        Z, Y = embed(torch.cat(fs)), torch.tensor(ys, device=DEV); D = Z.shape[1]
        g = torch.Generator().manual_seed(0); sync(); t0 = time.time()
        for c in range(lo, hi):
            Zc = Z[Y == c]
            if len(Zc): sph_kmeans(Zc.cpu(), K_CENTRES, g)        # code nhóm chạy k-means trên CPU
        z0 = Z[:512]; kappa = 16.0 / (1 - z0 @ z0.T)[torch.triu(torch.ones(len(z0), len(z0), dtype=torch.bool, device=DEV), 1)].median().item()
        sync(); t_km = time.time() - t0
        CEN[lab] = (F.normalize(torch.randn(sp["C"], K_CENTRES, D, device=DEV), dim=2), kappa)   # đủ C lớp như cuối chuỗi task
        SPH_ROWS.append(dict(dataset=ds, source=lab, feat_dim=D, task1_images=len(items),
            feat_time_task1_s=t_feat, kmeans_time_task1_s=t_km,
            est_build_all_tasks_s=t_feat * n_train_all / max(len(items), 1) + t_km * T_,
            centres_MB_all_classes=sp["C"] * K_CENTRES * D * 4 / 1024 ** 2))
        print(f"  Spherical {lab}: trích {t_feat:.1f}s, k-means {t_km:.2f}s cho Task 1", flush=True)

    # ---------------- (c) suy luận: GFLOPs, ms/ảnh, VRAM ----------------
    model.eval(); x1 = torch.randn(1, 3, 224, 224, device=DEV); xb = torch.randn(BATCH_INFER, 3, 224, 224, device=DEV)
    def cfg_apt(x): return model(x).argmax(1)
    def cfg_apt_stale(x):
        logit, f = model(x, return_feat=True); cen, kp = CEN["APT (stale)"]; return sph_fuse(logit, f, cen, kp)
    def make_frozen(lab, fwd):
        def run(x):
            logit = model(x); cen, kp = CEN[lab]
            with torch.autocast(DEV, dtype=torch.float16, enabled=(DEV == "cuda")): f = fwd(x).float()
            return sph_fuse(logit, f, cen, kp)
        return run
    configs = [("APT + KAC (baseline)", cfg_apt, 0.0), ("+ Spherical, APT (stale)", cfg_apt_stale, 0.0),
               ("+ Spherical, ViT-B/16 đóng băng", make_frozen("ViT-B/16 đóng băng", lambda x: VIT(x)), 0.0)] + \
              [(f"+ Spherical, {lab}", make_frozen(lab, (lambda x, n=n: get_swin(n)(x))),
                sum(p.numel() for p in get_swin(n).parameters()) / 1e6) for n, lab in SWIN.items()]
    with torch.no_grad():
        for name, fn, extra in configs:
            gf = gflops(lambda: fn(x1)); reset_peak()
            sec = time_fn(lambda: fn(xb)); vram = peak_gb()
            INF_ROWS.append(dict(dataset=ds, config=name, GFLOPs_per_img=gf, ms_per_img=1000 * sec / BATCH_INFER,
                                 img_per_s=BATCH_INFER / sec, peak_VRAM_infer_GB=vram, extra_backbone_params_M=extra,
                                 extra_trainable_params=0))
            print(f"  suy luận {name}: {gf:.2f} GFLOPs/ảnh, {1000 * sec / BATCH_INFER:.2f} ms/ảnh", flush=True)
    del model, opt; reset_peak()
