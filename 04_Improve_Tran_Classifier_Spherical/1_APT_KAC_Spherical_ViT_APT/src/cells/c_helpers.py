# ==============================================================================
# 4a. HÀM PHỤ (config cho repo, lệnh chạy, biến môi trường) - luôn chạy ô này
# ==============================================================================
import yaml
def make_cfg(ds, max_task=-1, tag=None):
    """Config của repo + rand_split=False (thứ tự task đã nằm trong nhãn) + kích thước task của nhóm."""
    cfg = yaml.safe_load(open(f"{REPO}/{HP[ds]['config']}"))
    sizes = DS_SPEC[ds]["sizes"]
    cfg.update(rand_split=False, first_split_size=sizes[0], other_split_size=sizes[1], max_task=max_task, validation=False)
    path = f"configs/_{tag or ds}.yaml"; yaml.safe_dump(cfg, open(f"{REPO}/{path}", "w")); return path

def cmd(ds, seed, out, config, epochs=None):
    h = HP[ds]
    return [sys.executable, "-u", "run.py", "--config", config, "--gpuid", "0", "--repeat", "1", "--overwrite", "0",
            "--learner_type", "prompt", "--learner_name", "APT_Learner", "--prompt_param", str(PROMPT_PARAM),
            "--lr", str(h["lr"]), "--seed", str(seed), "--ema_coeff", str(h["ema"]),
            "--schedule", str(epochs or h["epochs"]), "--batch_size", str(BATCH), "--log_dir", out]

def run_env(ds, seed, dumps, head="kac"):
    return dict(os.environ, APT_HEAD=head, APT_KAC_GRIDS=str(KAC_GRIDS), APT_DUMP=dumps, APT_DUMP_APT=APT_DUMP_APT,
                APT_SPLIT_DIR=split_dir(ds, PHASE, seed), **({"APT_HEAD_LR": str(HEAD_LR)} if (HEAD_LR and head == "kac") else {}))

