# ==============================================================================
# 3. TRÍCH ĐẶC TRƯNG BACKBONE ĐÓNG BĂNG (mỗi ảnh một lần, dùng chung cho mọi seed) + đo chi phí
# ==============================================================================
import time, torch, timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.flop_counter import FlopCounterMode

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("thiết bị:", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "")

class Imgs(Dataset):
    def __init__(self, src, tf): self.src, self.tf = src, tf
    def __len__(self): return len(self.src)
    def __getitem__(self, i):
        x = self.src[i]
        if isinstance(x, str):
            with Image.open(x) as im: return self.tf(im.convert("RGB"))
        return self.tf(Image.fromarray(x))                                   # CIFAR: mảng 32x32x3

MODELS = {}
def get_model(name):
    if name not in MODELS:
        for k in list(MODELS): del MODELS[k]                                  # giữ 1 model trên GPU
        torch.cuda.empty_cache() if DEV == "cuda" else None
        m = timm.create_model(name, pretrained=PRETRAINED, num_classes=0).eval().to(DEV)   # num_classes=0 -> vector sau pooling
        cfg = timm.data.resolve_data_config({}, model=m)
        MODELS[name] = (m, timm.data.create_transform(**cfg))
        print(f"{name}: {m.num_features} chiều | tiền xử lý {cfg['input_size']}, crop_pct {cfg.get('crop_pct')}")
    return MODELS[name]

@torch.no_grad()
def extract(name, src):
    m, tf = get_model(name); out, t0 = [], time.time()
    dl = DataLoader(Imgs(src, tf), batch_size=BATCH, shuffle=False, num_workers=WORKERS, pin_memory=(DEV == "cuda"))
    for b, x in enumerate(dl):
        with torch.autocast(DEV, dtype=torch.float16, enabled=(DEV == "cuda")):
            out.append(m(x.to(DEV, non_blocking=True)).float().cpu())
        if b % 50 == 0: print(f"   {min((b + 1) * BATCH, len(src))}/{len(src)} ảnh, {time.time() - t0:.0f}s", flush=True)
    return torch.cat(out).half()

FEAT = {}
def features(ds, name):
    """{"tr": tensor, "te": tensor}; cache ở OUT/feats để chạy lại không phải trích lần nữa."""
    if (ds, name) in FEAT: return FEAT[(ds, name)]
    S = load_source(ds); FEAT[(ds, name)] = {}
    for s in (["tr"] if PHASE == "val" else ["tr", "te"]):
        p = f"{OUT}/feats/{ds}_{s}_{name}.pt"
        if os.path.exists(p) and torch.load(p)["ids"] == S[s + "_id"]:
            FEAT[(ds, name)][s] = torch.load(p)["f"]; print(f"{ds}/{s}/{BACKBONES.get(name, name)}: dùng cache"); continue
        print(f"{ds}/{s}/{BACKBONES.get(name, name)}: trích {len(S[s + '_y'])} ảnh ...", flush=True)
        FEAT[(ds, name)][s] = extract(name, S[s]); os.makedirs(os.path.dirname(p), exist_ok=True)
        torch.save({"ids": S[s + "_id"], "f": FEAT[(ds, name)][s]}, p)
    return FEAT[(ds, name)]

@torch.no_grad()
def cost(name, label):
    """Số tham số, GFLOPs / ảnh 224x224, tốc độ forward thuần (fp16, batch BATCH, không tính đọc ảnh)."""
    m = timm.create_model(name, pretrained=False, num_classes=0).eval().to(DEV)
    x1 = torch.randn(1, 3, 224, 224, device=DEV)
    with FlopCounterMode(display=False) as fc: m(x1)
    x = torch.randn(BATCH, 3, 224, 224, device=DEV); n = 10 if DEV == "cuda" else 1
    with torch.autocast(DEV, dtype=torch.float16, enabled=(DEV == "cuda")):
        m(x)
        if DEV == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n): m(x)
        if DEV == "cuda": torch.cuda.synchronize()
    r = dict(backbone=label, timm=name, params_M=sum(p.numel() for p in m.parameters()) / 1e6,
             GFLOPs=fc.get_total_flops() / 2e9, img_per_s=n * BATCH / (time.time() - t0), dim=m.num_features)
    del m; torch.cuda.empty_cache() if DEV == "cuda" else None
    return r

COST = [cost("vit_base_patch16_224", "ViT-B/16 (nhánh ViT đóng băng)")] + [cost(n, l) for n, l in BACKBONES.items()]
import pandas as pd
print(pd.DataFrame(COST).round(2).to_string(index=False))
