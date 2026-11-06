"""Patch the official APT repo (in place): optional KAC head + dump of logits / frozen features after every task.
Env vars read at run time:  APT_HEAD=cosine|kac (default cosine = official),  APT_KAC_GRIDS (default 4),  APT_DUMP=<dir> (dump on),
APT_DUMP_APT=0|1|2 (also dump APT features; 2 = also re-embed old train images, reference only)."""
import os, re

def _sub(path, old, new, count=1):
    s = open(path).read()
    assert s.count(old) == count, f"{path}: anchor found {s.count(old)}x (expected {count}): {old[:60]!r}"
    open(path, "w").write(s.replace(old, new))

KAC_CLASS = '''
class KACHead(nn.Module):
    """KAC (CVPR'25): LayerNorm -> Gaussian RBF on evenly spaced grids in [-2, 2] -> linear without bias (zero init)."""
    def __init__(self, dim, n_cls, num_grids=4, grid_range=(-2.0, 2.0)):
        super().__init__(); gmin, gmax = grid_range
        self.layernorm = nn.LayerNorm(dim)
        self.register_buffer("grid", torch.linspace(gmin, gmax, num_grids))
        self.denominator = (gmax - gmin) / (num_grids - 1)
        self.spline_linear = nn.Linear(dim * num_grids, n_cls, bias=False); nn.init.zeros_(self.spline_linear.weight)
    def forward(self, f):
        x = self.layernorm(f)
        return self.spline_linear(torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2).flatten(1))

'''

DUMP_METHOD = '''
    @torch.no_grad()
    def _dump(self, i):
        """After task i: frozen (prompt-free) features of task-i TRAIN images, frozen features of task-i TEST images,
        and the model's logits (merged prompt, all seen classes) on the test images of tasks 0..i.
        APT_DUMP_APT=1|2 also dumps the APT features (CLS after the prompts = the vector the head sees):
          train_apt_{i}_{i}.pt  task-i train images, model after task i      (what a real continual run can store)
          test_apt_{i}_{j}.pt   task-j test images (j<=i), model after task i
        APT_DUMP_APT=2 adds train_apt_{i}_{j}.pt (j<i): OLD train images re-embedded by the current model. A real continual
        run does not have those images, so it is only a reference that separates feature drift from feature quality."""
        d = os.path.join(os.environ["APT_DUMP"], "seed%d" % self.seed); os.makedirs(d, exist_ok=True)
        mode = os.environ.get("APT_DUMP_APT", "0"); want_apt = mode != "0"
        model = self.learner.model; was_training = model.training; model.eval()
        valid = self.learner.valid_out_dim

        def run(ds, want_logits, want_apt, want_frozen=True):
            loader = DataLoader(ds, batch_size=128, shuffle=False, drop_last=False, num_workers=int(self.workers))
            F_, K_, A_, Y_ = [], [], [], []
            for x, y, _ in loader:
                x = x.cuda()
                if want_frozen: F_.append(model.feat(x)[:, 0, :].float().cpu())      # frozen ViT, no prompt
                if want_logits or want_apt:
                    o = model(x)
                    if want_logits: K_.append(o[:, :valid].float().cpu())
                    if want_apt:
                        assert hasattr(model, "feat_out"), "model has no feat_out (patch not applied?)"
                        A_.append(model.feat_out.float().cpu())                       # CLS after the prompts
                Y_.append(y)
            cat = lambda L: torch.cat(L) if L else None
            return cat(F_), cat(K_), cat(A_), torch.cat(Y_)

        tr_tf = self.train_dataset.transform
        self.train_dataset.transform = self.test_dataset.transform              # no augmentation for the prototypes
        for j in (range(i + 1) if mode == "2" else [i]):                        # ascending: task i is loaded last (as before)
            self.train_dataset.load_dataset(j, train=True)
            f, _, a, y = run(self.train_dataset, False, want_apt, want_frozen=(j == i))
            if j == i: torch.save({"f": f.half(), "y": y}, os.path.join(d, "train_frozen_%d.pt" % i))
            if want_apt: torch.save({"f": a.half(), "y": y}, os.path.join(d, "train_apt_%d_%d.pt" % (i, j)))
        self.train_dataset.transform = tr_tf
        for j in range(i + 1):
            self.test_dataset.load_dataset(j, train=True)
            f, k, a, y = run(self.test_dataset, True, want_apt)
            if j == i: torch.save({"f": f.half(), "y": y}, os.path.join(d, "test_frozen_%d.pt" % j))
            torch.save({"k": k.half(), "y": y}, os.path.join(d, "test_%d_%d.pt" % (i, j)))
            if want_apt: torch.save({"f": a.half(), "y": y}, os.path.join(d, "test_apt_%d_%d.pt" % (i, j)))
        model.train(was_training)

'''

def apply(repo):
    z = f"{repo}/models/zoo.py"; t = f"{repo}/trainer.py"; dl = f"{repo}/dataloaders/dataloader.py"; pl = f"{repo}/learners/prompt.py"
    if "APT_PATCHED" in open(z).read(): return "already patched"
    # lr riêng cho head (APT_HEAD_LR); không đặt thì giống repo: một lr chung
    _sub(pl, "        # create optimizers\n        self.optimizer = torch.optim.__dict__",
         "        if os.environ.get('APT_HEAD_LR'):\n"
         "            mod = self.model.module if len(self.config['gpuid']) > 1 else self.model\n"
         "            optimizer_arg['params'] = [{'params': list(mod.prompt.parameters()), 'lr': self.config['lr']},\n"
         "                                       {'params': list(mod.last.parameters()), 'lr': float(os.environ['APT_HEAD_LR'])}]\n"
         "        # create optimizers\n        self.optimizer = torch.optim.__dict__")
    _sub(pl, "from __future__ import print_function\n", "from __future__ import print_function\nimport os\n")
    # numpy>=2 (Colab): np.fromstring in binary mode was removed
    _sub(dl, "np.fromstring(image.tobytes(), dtype=np.uint8)", "np.frombuffer(image.tobytes(), dtype=np.uint8).copy()")
    _sub(z, "# note - ortho init has not been found", KAC_CLASS + "# APT_PATCHED\n# note - ortho init has not been found")
    _sub(z, "        self.last = nn.Linear(768, num_classes) \n",
         "        self.use_kac = os.environ.get('APT_HEAD', 'cosine') == 'kac'\n"
         "        self.last = KACHead(768, num_classes, int(os.environ.get('APT_KAC_GRIDS', 4))) if self.use_kac else nn.Linear(768, num_classes)\n")
    _sub(z, '            "last.weight",\n            "last.bias", \n            ] \n        else:',
         '            "last.weight",\n            "last.bias", \n            "last.layernorm.weight", "last.layernorm.bias", "last.spline_linear.weight",\n            ] \n        else:')
    _sub(z, '            "last.weight",\n            "last.bias", \n            ]\n\n        for name',
         '            "last.weight",\n            "last.bias", \n            "last.layernorm.weight", "last.layernorm.bias", "last.spline_linear.weight",\n            ]\n\n        for name')
    _sub(z, "        out = self.clf_norm(out)\n        wt_norm",
         "        self.feat_out = out.detach()\n        if self.use_kac:\n            return self.last(out)\n        out = self.clf_norm(out)\n        wt_norm")
    s = open(z).read()
    if not re.search(r"^import os\b", s, re.M): open(z, "w").write("import os\n" + s)
    _sub(t, "    def evaluate(self, avg_metrics):", DUMP_METHOD + "    def evaluate(self, avg_metrics):")
    _sub(t, "                acc_table.append(self.task_eval(j))\n            temp_table",
         "                acc_table.append(self.task_eval(j))\n            if os.environ.get('APT_DUMP'): self._dump(i)\n            temp_table")
    s = open(t).read()
    if not re.search(r"^import os\b", s, re.M): open(t, "w").write("import os\n" + s)
    return "patched"
