# ==============================================================================
# 6. BẢNG KẾT QUẢ (chọn cấu hình theo Last trung bình các seed trên Val, như các notebook trước)
# ==============================================================================
CFG = ["w", "kappa", "k", "lam", "w_in"]
fmt = lambda m, s: f"{m:6.2f}" + (f" ± {s:4.2f}" if pd.notna(s) else "       ")

def best(df, keys):
    df = df.copy(); df.loc[(df.variant == "fuse") & (df.w == 1.0), "variant"] = "proto"
    g = df.groupby(keys + ["variant"] + CFG, dropna=False).agg(n=("seed", "nunique"), avg=("avg", "mean"),
        last=("last", "mean"), last_sd=("last", "std"), forg=("forgetting", "mean")).reset_index()
    return g.loc[g.groupby(keys + ["variant"], dropna=False)["last"].idxmax()]

R1 = pd.DataFrame(ROWS1); R1F = pd.DataFrame(ROWS1F); R2 = pd.DataFrame(ROWS2)
OUTS = []
for ds in DATASETS:
    if ds not in R1.dataset.values: continue
    print(f"\n==================== {ds} ====================")
    print("[1] KAC hiệu chỉnh — không nhánh phụ, 1 lượt forward")
    a = R1[R1.dataset == ds].groupby("calib").agg(n=("seed", "nunique"), avg=("avg", "mean"), last=("last", "mean"),
        last_sd=("last", "std"), forg=("forgetting", "mean"), task_id=("task_id_last", "mean")).reindex(CALIB)
    for h, r in a.iterrows():
        print(f"   {h:14} Avg {r.avg:6.2f} | Last {fmt(r['last'], r.last_sd)} | Forg {r.forg:5.2f} | chọn đúng task (sau task cuối) {r.task_id:5.2f}")
    a.insert(0, "dataset", ds); OUTS.append(a.reset_index().assign(table="1_kac_calib"))

    print("[1b] KAC hiệu chỉnh + Spherical-ViT (2 lượt forward như cũ)")
    base = R2[(R2.dataset == ds) & (R2.proj == "full")].assign(kac="raw")
    b = best(pd.concat([base, R1F[R1F.dataset == ds]]), ["kac"])
    for _, r in b.sort_values(["variant", "kac"]).iterrows():
        cfg = " ".join(f"{c}={r[c]:g}" for c in CFG if pd.notna(r[c]))
        print(f"   {r.variant:5} KAC={r.kac:14} Last {fmt(r['last'], r.last_sd)} | Forg {r.forg:5.2f} | {cfg}")
    OUTS.append(b.assign(dataset=ds, table="1b_calib_fuse"))

    print("[2] Giảm chiều Spherical-ViT (KAC gốc)")
    c = best(R2[R2.dataset == ds], ["proj", "dim"])
    piv = c.pivot_table(index=["proj", "dim"], columns="variant", values="last", dropna=False)
    sd = c.pivot_table(index=["proj", "dim"], columns="variant", values="last_sd", dropna=False)
    order = [("full", 768)] + [(p, d) for p in PROJ_TYPES for d in sorted(PROJ_DIMS, reverse=True)]
    for p, d in order:
        if (p, d) not in piv.index: continue
        kk = int(c[(c.proj == p) & (c.dim == d) & (c.variant == "fuse")]["k"].iloc[0])
        print(f"   {p:4} d={d:4} | proto {fmt(piv.loc[(p, d), 'proto'], sd.loc[(p, d), 'proto'])} | fuse {fmt(piv.loc[(p, d), 'fuse'], sd.loc[(p, d), 'fuse'])}"
              f" | hier {fmt(piv.loc[(p, d), 'hier'], sd.loc[(p, d), 'hier'])} | bộ nhớ tâm lớp (fuse, k={kk}) ~{kk * d} số/lớp")
    OUTS.append(c.assign(dataset=ds, table="2_projection"))

pd.concat(OUTS, ignore_index=True).to_csv(f"{OUT}/exp_summary_{PHASE}.csv", index=False, float_format="%.3f")
pd.DataFrame(EVR).groupby(["dataset", "proj", "dim"]).explained_var_task1.mean().dropna().round(3).to_csv(f"{OUT}/pca_explained_var.csv")
print("\nPCA: tỉ lệ phương sai giữ lại (trên ảnh train task 1):")
print(pd.DataFrame(EVR).dropna().groupby(["dataset", "dim"]).explained_var_task1.mean().round(3).unstack())
