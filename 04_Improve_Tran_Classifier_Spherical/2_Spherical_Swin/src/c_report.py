# ==============================================================================
# 6. BẢNG KẾT QUẢ: chọn cấu hình theo Last trung bình các seed (trên Val), ViT và Swin cạnh nhau
# ==============================================================================
df = pd.DataFrame(ROWS)
df.loc[(df.variant == "fuse") & (df.w == 1.0), "variant"] = "proto"       # fuse với w = 1 chính là Spherical thuần
df.loc[df.variant.isin(["KAC", "oracle"]), "backbone"] = "-"               # KAC/oracle không phụ thuộc backbone Spherical
df = df.drop_duplicates(subset=["dataset", "seed", "backbone", "variant", "w", "kappa", "k", "lam", "w_in"])
CFG = ["w", "kappa", "k", "lam", "w_in"]
ORDER = [("-", "KAC")] + [(b, v) for b in ["ViT"] + list(BACKBONES.values()) for v in ("proto", "fuse", "hier")] + [("-", "oracle")]

def best_table(sub):
    g = sub.groupby(["backbone", "variant"] + CFG, dropna=False).agg(
        n=("seed", "nunique"), seeds=("seed", lambda s: ",".join(map(str, sorted(s.unique())))),
        avg=("avg", "mean"), avg_sd=("avg", "std"), last=("last", "mean"), last_sd=("last", "std"),
        forg=("forgetting", "mean"), forg_sd=("forgetting", "std")).reset_index()
    out = []
    for bb, v in ORDER:
        x = g[(g.backbone == bb) & (g.variant == v)]
        if len(x): out.append(x.loc[x["last"].idxmax()])
    return pd.DataFrame(out)

fmt = lambda m, s: f"{m:.2f}" + (f" ± {s:.2f}" if pd.notna(s) else "")
TABLES = []
for ds in DATASETS:
    sub = df[df.dataset == ds]
    paired = sub[sub.has_kac]
    if len(paired):
        t = best_table(paired); t.insert(0, "dataset", ds); t.insert(1, "nhóm", "có KAC (so sánh cặp)"); TABLES.append(t)
        print(f"\n### {ds} — các seed có dump KAC: {sorted(int(s) for s in paired.seed.unique())}")
        for _, r in t.iterrows():
            cfg = " ".join(f"{c}={r[c]:g}" for c in CFG if pd.notna(r[c]) and not (r.variant == "proto" and c == "w"))
            print(f"  {r.backbone:7}{r.variant:7} Avg {fmt(r.avg, r.avg_sd):15} Last {fmt(r['last'], r.last_sd):15} Forg {fmt(r.forg, r.forg_sd):14} {cfg}")
    if len(paired) == 0 or set(sub.seed) != set(paired.seed):              # có seed thiếu dump: báo riêng proto trên mọi seed
        for bb in BACKBONES.values():
            t2 = best_table(sub[(sub.backbone == bb) & (sub.variant == "proto")]); t2.insert(0, "dataset", ds)
            t2.insert(1, "nhóm", f"{bb} proto, mọi seed"); TABLES.append(t2); r = t2.iloc[0]
            print(f"  {bb} proto trên mọi seed ({r.seeds}): Avg {fmt(r.avg, r.avg_sd)} | Last {fmt(r['last'], r.last_sd)} | Forg {fmt(r.forg, r.forg_sd)}")

dg = pd.DataFrame(DIAG).groupby(["dataset", "backbone"]).task_id_acc_old_last.agg(["mean", "std", "count"]).round(2)
print("\nĐộ chính xác chọn đúng task của Spherical (k=1, κ_mult=16; task cũ, sau task cuối):\n", dg)
pd.concat(TABLES).to_csv(f"{OUT}/swin_vs_vit_best_{PHASE}.csv", index=False, float_format="%.4f")
dg.to_csv(f"{OUT}/task_id_acc_{PHASE}.csv")
print("\nĐã lưu:", f"{OUT}/swin_vs_vit_best_{PHASE}.csv", f"{OUT}/rows_all_{PHASE}.csv", f"{OUT}/task_id_acc_{PHASE}.csv")

pd.DataFrame(COST).to_csv(f"{OUT}/backbone_cost.csv", index=False, float_format="%.2f")
print("\nChi phí backbone (forward thuần, fp16):\n", pd.DataFrame(COST).round(2).to_string(index=False))
