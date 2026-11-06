# ==============================================================================
# 3. MÔ HÌNH: ViT-B/16 đóng băng + prompt cộng vào CLS (APT) + head KAC, và nhánh Spherical
#    Kiến trúc giống repo APT (24 prompt = 12 tầng x {K, V}, cộng vào token CLS) và KAC (4 RBF).
# ==============================================================================
import time, math, json
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F, timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.flop_counter import FlopCounterMode
import torchvision.transforms as T

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GPU_NAME = torch.cuda.get_device_name(0) if DEV == "cuda" else "CPU"
print("thiết bị:", GPU_NAME)
if DEV == "cuda":
    torch.backends.cudnn.benchmark = True

class Attention(nn.Module):
    def __init__(self, dim=768, heads=12):
        super().__init__(); self.heads, self.hd = heads, dim // heads
        self.qkv = nn.Linear(dim, dim * 3); self.proj = nn.Linear(dim, dim)
    def forward(self, x, prompt=None):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4).unbind(0)
        if prompt is not None:
            pk, pv = prompt
            k = torch.cat([k[:, :, :1] + pk.to(k.dtype), k[:, :, 1:]], dim=2)
            v = torch.cat([v[:, :, :1] + pv.to(v.dtype), v[:, :, 1:]], dim=2)
        return self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C))

class Block(nn.Module):
    def __init__(self, dim=768, heads=12):
        super().__init__()
        self.norm1, self.attn = nn.LayerNorm(dim, eps=1e-6), Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
    def forward(self, x, prompt=None):
        x = x + self.attn(self.norm1(x), prompt)
        return x + self.mlp(self.norm2(x))

class ViT(nn.Module):
    def __init__(self, dim=768, depth=12, heads=12):
        super().__init__(); self.depth, self.heads = depth, heads
        self.patch_embed = nn.Conv2d(3, dim, 16, 16)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim)); self.pos_embed = nn.Parameter(torch.zeros(1, 197, dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)]); self.norm = nn.LayerNorm(dim, eps=1e-6)
    def forward(self, x, prompt_fn=None):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], 1) + self.pos_embed
        for i, b in enumerate(self.blocks): x = b(x, prompt_fn(i) if prompt_fn else None)
        return self.norm(x)[:, 0]

def make_vit():
    vit = ViT()
    if PRETRAINED:                                   # chỉ cần khi muốn trọng số thật (không ảnh hưởng chi phí)
        ref = timm.create_model(VIT_NAME, pretrained=True, num_classes=0)
        sd = {k.replace("mlp.fc1", "mlp.0").replace("mlp.fc2", "mlp.2").replace("patch_embed.proj", "patch_embed"): v
              for k, v in ref.state_dict().items() if not k.startswith("head")}
        print(vit.load_state_dict(sd, strict=False)); del ref
    for p in vit.parameters(): p.requires_grad_(False)
    return vit.eval()

class APTPrompt(nn.Module):
    def __init__(self, depth=12, dim=768, heads=12, ema=0.7):
        super().__init__(); self.heads, self.hd, self.ema = heads, dim // heads, ema
        self.prompt_tokens = nn.Parameter(torch.randn(depth * 2, dim) * 0.02)
        self.register_buffer("merged", torch.zeros(depth * 2, dim)); self.n_merged = 0; self.use_merged = False
    def get(self, l):
        P = self.merged if self.use_merged else self.prompt_tokens
        return P[2 * l].view(self.heads, 1, self.hd), P[2 * l + 1].view(self.heads, 1, self.hd)
    @torch.no_grad()
    def merge(self):
        self.merged.copy_(self.prompt_tokens if self.n_merged == 0 else self.merged * self.ema + self.prompt_tokens * (1 - self.ema))
        self.n_merged += 1

class KACHead(nn.Module):
    def __init__(self, dim, n_cls, num_grids=4, grid_range=(-2.0, 2.0)):
        super().__init__(); gmin, gmax = grid_range
        self.layernorm = nn.LayerNorm(dim); self.register_buffer("grid", torch.linspace(gmin, gmax, num_grids))
        self.den = (gmax - gmin) / (num_grids - 1)
        self.spline_linear = nn.Linear(dim * num_grids, n_cls, bias=False); nn.init.zeros_(self.spline_linear.weight)
    def forward(self, f):
        x = self.layernorm(f)
        return self.spline_linear(torch.exp(-((x[..., None] - self.grid) / self.den) ** 2).flatten(1))

class APTKAC(nn.Module):
    def __init__(self, vit, n_cls):
        super().__init__(); self.vit, self.prompt, self.head = vit, APTPrompt(), KACHead(768, n_cls)
    def features(self, x):
        self.prompt.use_merged = (not self.training) and self.prompt.n_merged > 0
        with torch.autocast(DEV, dtype=torch.float16, enabled=(DEV == "cuda")):
            return self.vit(x, prompt_fn=self.prompt.get).float()
    def forward(self, x, return_feat=False):
        f = self.features(x); logit = self.head(f)
        return (logit, f) if return_feat else logit

# ---------------- nhánh Spherical (giống fusion_online.py của nhóm) ----------------
def embed(f): return F.normalize(F.layer_norm(f.float(), (f.shape[1],)), dim=1)

def sph_kmeans(Z, k, g, iters=20):
    k = min(k, Z.shape[0]); C = Z[torch.randperm(Z.shape[0], generator=g)[:k]].clone()
    for _ in range(iters):
        a = (Z @ C.T).argmax(1); newC = torch.zeros_like(C).index_add_(0, a, Z)
        empty = newC.norm(dim=1) < 1e-8
        if empty.any(): newC[empty] = Z[torch.randint(0, Z.shape[0], (int(empty.sum()),), generator=g)]
        C = F.normalize(newC, dim=1)
    return C

def sph_fuse(logit, f, cen, kappa, w=0.5):
    """fuse: (1-w) log P_KAC + w log P_sph trên mọi lớp đã học; cen: (C, k, D)."""
    z = embed(f); cos = torch.einsum("nd,ckd->nck", z, cen)
    s = torch.logsumexp(kappa * (cos - 1), dim=2)
    return ((1 - w) * F.log_softmax(logit.float(), 1) + w * F.log_softmax(s, 1)).argmax(1)

class Imgs(Dataset):
    def __init__(self, src, ys, tf): self.src, self.ys, self.tf = src, ys, tf
    def __len__(self): return len(self.src)
    def __getitem__(self, i):
        x = self.src[i]
        if isinstance(x, str):
            with Image.open(x) as im: x = im.convert("RGB")
        else: x = Image.fromarray(x)
        return self.tf(x), self.ys[i]

TRAIN_TF = T.Compose([T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ToTensor()])
TEST_TF = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

def sync():
    if DEV == "cuda": torch.cuda.synchronize()

def time_fn(fn):
    """giây / lần gọi (đo GPU thuần, dữ liệu đã nằm sẵn trên GPU)."""
    for _ in range(N_WARM): fn()
    sync(); t0 = time.time()
    for _ in range(N_REP): fn()
    sync(); return (time.time() - t0) / N_REP

def reset_peak():
    if DEV == "cuda": torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
def peak_gb(): return torch.cuda.max_memory_allocated() / 1024 ** 3 if DEV == "cuda" else float("nan")

from torch.nn.attention import sdpa_kernel, SDPBackend
def gflops(fn):
    # ép attention chạy đường MATH để FlopCounter đếm cả phép nhân QK^T và AV (kernel fused có thể không được đếm)
    with sdpa_kernel(SDPBackend.MATH), FlopCounterMode(display=False) as fc: fn()
    return fc.get_total_flops() / 2e9          # FLOPs/2 = MACs, cùng quy ước với bảng backbone (17.56 G cho ViT-B)
