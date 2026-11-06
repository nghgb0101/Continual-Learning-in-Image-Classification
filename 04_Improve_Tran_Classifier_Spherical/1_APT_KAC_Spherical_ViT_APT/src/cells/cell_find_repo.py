import os, sys, glob, shutil, subprocess, zipfile
os.makedirs(f"{DRIVE_DIR}/runs", exist_ok=True)
INPUT_ROOT = "/kaggle/input"
print("Nội dung /kaggle/input:", os.listdir(INPUT_ROOT) if os.path.isdir(INPUT_ROOT) else "KHÔNG CÓ")
if not glob.glob("/tmp/apt_repo/**/run.py", recursive=True):
    runs = glob.glob(f"{INPUT_ROOT}/**/run.py", recursive=True)          # Kaggle thường tự giải nén zip khi upload
    zips = glob.glob(f"{INPUT_ROOT}/**/RepoGit_APT.zip", recursive=True)
    assert runs or zips, "Notebook chưa có dataset chứa repo: bên phải -> Add Input -> chọn dataset đã upload RepoGit_APT.zip"
    os.makedirs("/tmp/apt_repo", exist_ok=True)
    if zips: zipfile.ZipFile(zips[0]).extractall("/tmp/apt_repo")
    else: shutil.copytree(os.path.dirname(runs[0]), "/tmp/apt_repo/Additive-Prompt-Tuning-main")   # /kaggle/input chỉ đọc -> chép ra /tmp để patch
REPO = os.path.dirname(glob.glob("/tmp/apt_repo/**/run.py", recursive=True)[0])
print("Repo:", REPO)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "timm==0.4.12"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyaml", "gdown"], check=True)
import torch
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "KHÔNG CÓ GPU")
assert torch.cuda.is_available(), "Bật GPU: Settings -> Accelerator -> GPU T4 x2 (hoặc P100)"
r = subprocess.run([sys.executable, "-c", """
import sys; sys.path.insert(0,'.')
import timm, torch
from timm.models import vit_base_patch16_224
import timm.models.vision_transformer as v
from models.vit import VisionTransformer
print('timm', timm.__version__); print(v.default_cfgs['vit_base_patch16_224']['url'])
m = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, ckpt_layer=0, drop_path_rate=0)
sd = vit_base_patch16_224(pretrained=True).state_dict(); del sd['head.weight']; del sd['head.bias']
r = m.load_state_dict(sd, strict=False); print('missing', r.missing_keys, 'unexpected', r.unexpected_keys)
assert not r.missing_keys and not r.unexpected_keys
print('OK trọng số khớp')
"""], cwd=REPO, capture_output=True, text=True)
print(r.stdout[-600:], r.stderr[-800:] if r.returncode else "")
assert r.returncode == 0, "Cần bật Internet: Settings -> Internet On"
