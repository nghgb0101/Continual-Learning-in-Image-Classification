"""Patch dataloaders/dataloader.py of the official APT repo: when env APT_SPLIT_DIR is set, CIFAR100 / CUB200 /
StanfordCars / ImageNet_R read their data from {APT_SPLIT_DIR}/train.json or test.json (written by the data cell).
Without APT_SPLIT_DIR the repo behaves exactly as before."""

SPLIT_PATCH = r'''

# ===== APT_SPLIT_PATCH: đọc split train/val/test của nhóm từ $APT_SPLIT_DIR/{train,test}.json =====
import json as _json
_CIFAR_CACHE = {}
def _apt_split_load(self):
    d = os.environ["APT_SPLIT_DIR"]
    js = _json.load(open(os.path.join(d, "train.json" if (self.train or self.validation) else "test.json")))
    self.targets = list(js["targets"])
    if js["kind"] == "cifar":
        out = []
        for item in js["data"]:
            split, i = item.split(":")
            if (js["root"], split) not in _CIFAR_CACHE:
                with open(os.path.join(js["root"], split), "rb") as f:
                    entry = pickle.load(f, encoding="latin1")
                _CIFAR_CACHE[(js["root"], split)] = entry["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            out.append(_CIFAR_CACHE[(js["root"], split)][int(i)])
        self.data = np.stack(out)
    else:
        self.data = list(js["data"])

def _apt_wrap_load(cls):
    orig = cls.load
    def load(self):
        if os.environ.get("APT_SPLIT_DIR"):
            return _apt_split_load(self)
        return orig(self)
    cls.load = load

for _cls in (iCIFAR10, iIMAGENET_R, iCUB200, iStanfordCars):   # iCIFAR100 kế thừa load của iCIFAR10
    _apt_wrap_load(_cls)
'''


def apply_split_patch(repo):
    p = f"{repo}/dataloaders/dataloader.py"
    s = open(p).read()
    if "APT_SPLIT_PATCH" in s:
        return "split patch: already applied"
    for name in ("class iCIFAR10(", "class iCIFAR100(", "class iIMAGENET_R(", "class iCUB200(", "class iStanfordCars("):
        assert name in s, f"dataloader.py: không thấy {name}"
    assert "def load(self)" in s.split("class iCIFAR100(")[0].split("class iCIFAR10(")[1], "iCIFAR10.load không như mong đợi"
    assert "def load" not in s.split("class iCIFAR100(")[1].split("class iIMAGENET_R(")[0], "iCIFAR100 có load riêng - cần patch thêm"
    open(p, "w").write(s + SPLIT_PATCH)
    return "split patch: applied"
