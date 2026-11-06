# ==============================================================================
# 6. PHÂN TÍCH: PHASE="val" -> chọn cấu hình (theo Last trung bình 3 seed) và lưu best_configs_val.json
#               PHASE="test" -> dùng đúng cấu hình đã chọn ở "val", báo cáo trên Test
# Gom dump của mọi seed: phiên hiện tại + output của phiên trước (Add Input -> notebook output).
# ==============================================================================
import json, pandas as pd
def gather_dumps(ds, phase):
    dst = f"/kaggle/working/_merged/{ds}/{phase}"; shutil.rmtree(dst, ignore_errors=True); os.makedirs(dst)
    for pat in (f"{DRIVE_DIR}/runs/kac/{ds}/{phase}/dumps", f"/kaggle/input/**/runs/kac/{ds}/{phase}/dumps"):
        for parent in sorted(glob.glob(pat, recursive=True)):
            for s, (d, T) in seeds_complete(parent).items():
                if s in SEEDS and not os.path.exists(f"{dst}/seed{s}"): os.symlink(os.path.abspath(d), f"{dst}/seed{s}")
    return dst

KEYS = ["variant", "w", "kappa", "k", "lam", "w_in"]
VARIANTS = {"proto": ("fuse", {"w": 1.0}), "fuse": ("fuse", {}), "hier": ("hier", {})}
ROWS, DIAG, AG = {}, {}, {}
for ds in DATASETS:
    dd = gather_dumps(ds, PHASE)
    print(f"===== {ds} | {PHASE} | seed có đủ dump: {sorted(os.listdir(dd))}", flush=True)
    if not os.listdir(dd): continue
    ROWS[ds], DIAG[ds] = analyse_all(dd)
    d = pd.DataFrame(ROWS[ds])
    AG[ds] = d.groupby(["source"] + KEYS, dropna=False).agg(seeds=("seed", "nunique"), avg=("avg", "mean"), avg_sd=("avg", "std"),
        last=("last", "mean"), last_sd=("last", "std"), forg=("forgetting", "mean"), forg_sd=("forgetting", "std")).reset_index()
    AG[ds].assign(dataset=ds, phase=PHASE).to_csv(f"{DRIVE_DIR}/fusion_apt_summary_{ds}_{PHASE}.csv", index=False)

clean = lambda r: {k: (None if pd.isna(r[k]) else (float(r[k]) if k != "variant" else r[k])) for k in KEYS}
BEST_PATH = f"{DRIVE_DIR}/best_configs_val.json"
def load_best():
    """Gộp best_configs_val.json của mọi phiên PHASE="val" (Add Input output các phiên trước) + phiên hiện tại."""
    best = {}
    for p in sorted(glob.glob("/kaggle/input/**/best_configs_val.json", recursive=True)) + [BEST_PATH]:
        if os.path.exists(p): best.update(json.load(open(p))); print("  đọc cấu hình:", p)
    return best
BEST = load_best()
if PHASE == "val":
    for ds, ag in AG.items():
        BEST[ds] = {}
        for src in [s for s in SOURCES if s in set(ag.source)]:
            BEST[ds][src] = {}
            for v, (vv, cond) in VARIANTS.items():
                m = ag[(ag.source == src) & (ag.variant == vv)]
                for a, b in cond.items(): m = m[m[a] == b]
                BEST[ds][src][v] = clean(m.sort_values("last", ascending=False).iloc[0])
    json.dump(BEST, open(BEST_PATH, "w"), indent=1); print("Đã lưu cấu hình chọn trên Val:", BEST_PATH, "| dataset:", sorted(BEST))
else:
    miss = [ds for ds in AG if ds not in BEST]
    assert not miss, f"PHASE='test' cần cấu hình chọn trên Val cho {miss}: Add Input output của notebook chạy PHASE='val' (có best_configs_val.json)"

def pick(ag, src, cfg):
    m = ag[ag.source == src]
    for k in KEYS: m = m[m[k].isna()] if cfg[k] is None else m[m[k] == cfg[k]]
    return m.iloc[0]
REPORT = []
for ds, ag in AG.items():
    if ds not in BEST: print(ds, ": chưa có cấu hình chọn trên Val -> bỏ qua"); continue
    base = ag[ag.source == "frozen"]
    for name in ("KAC", "oracle"):
        r = base[base.variant == name].iloc[0]; REPORT.append(dict(dataset=ds, source="-", method=name, **{c: r[c] for c in ("seeds", "avg", "avg_sd", "last", "last_sd", "forg", "forg_sd")}, config=""))
    for src, vs in BEST[ds].items():
        if src not in set(ag.source): continue
        for v, cfg in vs.items():
            r = pick(ag, src, cfg)
            REPORT.append(dict(dataset=ds, source=src, method=v, **{c: r[c] for c in ("seeds", "avg", "avg_sd", "last", "last_sd", "forg", "forg_sd")},
                               config=", ".join(f"{k}={cfg[k]:g}" for k in KEYS[1:] if cfg[k] is not None)))
REP = pd.DataFrame(REPORT)
if len(REP):
    REP.to_csv(f"{DRIVE_DIR}/report_{PHASE}.csv", index=False)
    fmt = lambda r: f"{r.avg:6.2f}±{r.avg_sd:4.2f} | {r['last']:6.2f}±{r.last_sd:4.2f} | {r.forg:6.2f}±{r.forg_sd:4.2f}"
    for ds in REP.dataset.unique():
        print(f"\n===== {ds} | {PHASE} | Avg | Last | Forgetting =====")
        for _, r in REP[REP.dataset == ds].iterrows(): print(f"  {r.source:12} {r.method:7}", fmt(r.fillna(0)), f"  [{r.config}]" if r.config else "")
        dg = pd.DataFrame(DIAG[ds]).groupby("source").mean(numeric_only=True).drop(columns="seed")
        print(dg.rename(columns={"task_id_acc_old_last": "chọn đúng task (%)", "cos_same_image_old_task_last": "cos cùng ảnh (drift)"}).round(3))
