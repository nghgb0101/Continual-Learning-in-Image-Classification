import re
"""Sinh các bảng LaTeX cho report Spherical từ CSV kết quả (Val, 3 seed)."""
import csv, statistics as st, sys, os

S = "/tmp/claude-0/-home-claude/01267e16-424c-53ad-9968-5cb515f47dea/scratchpad"
OUT = os.path.join(S, "rep/build")
D = list(csv.DictReader(open(f"{S}/gs/detail.csv")))
P = list(csv.DictReader(open(f"{S}/gs/perseed.csv")))
E = list(csv.DictReader(open(f"{S}/expres/exp_summary_val.csv")))
V = list(csv.DictReader(open(f"{S}/expres/pca_explained_var.csv")))

DS = ["CIFAR100", "CUB200", "StanfordCars", "ImageNet_R"]
DSN = {"CIFAR100": "CIFAR-100", "CUB200": "CUB-200", "StanfordCars": "Stanford Cars", "ImageNet_R": "ImageNet-R"}
SRC = ["frozen ViT-B/16", "apt_stale", "frozen Swin-B", "frozen Swin-T", "apt_refresh"]
SRCN = {"KAC": "APT + KAC", "frozen ViT-B/16": "ViT-B/16 đóng băng", "apt_stale": "APT (stale)",
        "apt_refresh": "APT (refresh)", "frozen Swin-B": "Swin-B đóng băng", "frozen Swin-T": "Swin-T đóng băng"}


def g(ds, src, var):
    r = [x for x in D if x["dataset"] == ds and x["source"] == src and x["variant"] == var]
    assert len(r) == 1, (ds, src, var)
    return r[0]


def pm(m, s, b=False):
    t = f"{float(m):.2f} $\\pm$ {float(s):.2f}"
    return f"\\textbf{{{t}}}" if b else t


def cfg(c):
    c = c.replace("kappa=", "$\\kappa$=").replace("lam=", "$\\lambda$=").replace("w_in=", "$w_{in}$=")
    return c.replace(" ", ", ")


out = {}

# ---------- Bảng 1: Last, biến thể fuse ----------
rows = []
for ds in DS:
    b = float(g(ds, "KAC", "KAC")["last"])
    vals = {s: g(ds, s, "fuse") for s in SRC}
    best = max(float(vals[s]["last"]) for s in SRC[:4])
    cells = [pm(g(ds, "KAC", "KAC")["last"], g(ds, "KAC", "KAC")["last_sd"])]
    for s in SRC:
        v = vals[s]
        cells.append(pm(v["last"], v["last_sd"], s != "apt_refresh" and float(v["last"]) == best))
    d = float(vals["frozen ViT-B/16"]["last"]) - b
    cells.append(f"{d:+.2f}")
    rows.append(DSN[ds] + " & " + " & ".join(cells) + r" \\")
out["t1"] = "\n".join(rows)

# ---------- Bảng 2: đầy đủ ----------
rows = []
for i, ds in enumerate(DS):
    blk = [("KAC", "KAC")] + [(s, v) for s in SRC for v in ("fuse", "hier")]
    n = len(blk)
    for j, (s, v) in enumerate(blk):
        r = g(ds, s, v)
        first = f"\\multirow{{{n}}}{{*}}{{{DSN[ds]}}}" if j == 0 else ""
        tag = "Baseline" if s == "KAC" else ("Tham chiếu" if s == "apt_refresh" else "Ours")
        vn = "--" if v == "KAC" else v
        rows.append(f"{first} & \\src{{{tag}}} {SRCN[s]} & {vn} & {pm(r['last'], r['last_sd'])} & "
                    f"{pm(r['avg'], r['avg_sd'])} & {pm(r['forg'], r['forg_sd'])} & {{\\scriptsize {cfg(r['config']) or '--'}}} \\\\")
        if j == 0 or (j % 2 == 0 and j < n - 1):
            rows.append(r"\cmidrule(l){2-7}")
    if i < len(DS) - 1:
        rows.append(r"\midrule")
out["t2"] = "\n".join(rows)


# ---------- Bảng 4a: calibration trên KAC ----------
def e(table, ds, **kw):
    r = [x for x in E if x["table"] == table and x["dataset"] == ds and all(x[k] == v for k, v in kw.items())]
    assert len(r) == 1, (table, ds, kw, len(r))
    return r[0]


CAL = [("raw", "Gốc (không hiệu chỉnh)"), ("center", "center"), ("center_half", "center\\_half"),
       ("zscore", "zscore"), ("block_softmax", "block\\_softmax")]
rows = []
for c, name in CAL:
    cells = []
    for ds in DS:
        r = e("1_kac_calib", ds, calib=c)
        base = float(e("1_kac_calib", ds, calib="raw")["last"])
        d = float(r["last"]) - base
        cells.append(f"{float(r['last']):.2f}" + ("" if c == "raw" else f" ({d:+.2f})"))
    rows.append(name + " & " + " & ".join(cells) + r" \\")
out["t4a"] = "\n".join(rows)

rows = []
for c, name in CAL:
    if c in ("center_half", "zscore"):
        continue
    cells = []
    for ds in DS:
        r = e("1b_calib_fuse", ds, kac=c, variant="fuse")
        base = float(e("1b_calib_fuse", ds, kac="raw", variant="fuse")["last"])
        d = float(r["last"]) - base
        cells.append(f"{float(r['last']):.2f}" + ("" if c == "raw" else f" ({d:+.2f})"))
    rows.append(name + " & " + " & ".join(cells) + r" \\")
out["t4b"] = "\n".join(rows)

# ---------- Bảng 5: giảm chiều (fuse) ----------
rows = []
for proj, dims in (("full", ["768.000"]), ("pca", ["256.000", "128.000", "64.000"]), ("rp", ["256.000", "128.000", "64.000"])):
    for dm in dims:
        cells = []
        for ds in DS:
            r = e("2_projection", ds, proj=proj, dim=dm, variant="fuse")
            base = float(e("2_projection", ds, proj="full", dim="768.000", variant="fuse")["last"])
            d = float(r["last"]) - base
            cells.append(f"{float(r['last']):.2f}" + ("" if proj == "full" else f" ({d:+.2f})"))
        ev = ""
        if proj == "pca":
            ev = " / ".join(f"{float([x for x in V if x['dataset'] == ds and x['dim'] == str(int(float(dm)))][0]['explained_var_task1']) * 100:.0f}" for ds in DS)
        name = {"full": "Đầy đủ", "pca": "PCA", "rp": "Chiếu ngẫu nhiên"}[proj]
        rows.append(f"{name} & {int(float(dm))} & " + " & ".join(cells) + f" & {ev or '--'} \\\\")
out["t5"] = "\n".join(rows)

# ---------- Phụ lục B: theo seed ----------
rows = []
for i, ds in enumerate(DS):
    rs = [x for x in P if x["dataset"] == ds]
    for j, r in enumerate(rs):
        s = [float(r[k]) for k in ("s0", "s1", "s2")]
        first = f"\\multirow{{{len(rs)}}}{{*}}{{{DSN[ds]}}}" if j == 0 else ""
        vn = "--" if r["variant"] == "KAC" else r["variant"]
        rows.append(f"{first} & {SRCN[r['source']]} & {vn} & " + " & ".join(f"{x:.2f}" for x in s)
                    + f" & {st.mean(s):.2f} $\\pm$ {st.stdev(s):.2f} \\\\")
    if i < len(DS) - 1:
        rows.append(r"\midrule")
out["tB"] = "\n".join(rows)

# đối chiếu: mean theo seed phải khớp detail.csv
for r in P:
    m = st.mean(float(r[k]) for k in ("s0", "s1", "s2"))
    d = float(g(r["dataset"], r["source"], r["variant"])["last"])
    assert abs(m - d) < 0.006, (r["dataset"], r["source"], r["variant"], m, d)

for k, v in out.items():
    v = re.sub(r"\((-)(\d)", r"(−\2", v); v = re.sub(r"^(.*?& )-(\d)", r"\1−\2", v, flags=re.M) if k=="t1" else v
    open(os.path.join(OUT, f"{k}.tex"), "w").write(v + "\n")
print("ok", list(out))
