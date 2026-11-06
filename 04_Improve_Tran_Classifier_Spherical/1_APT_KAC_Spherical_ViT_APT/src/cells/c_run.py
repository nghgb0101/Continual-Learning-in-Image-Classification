# ==============================================================================
# 5. CHẠY THẬT: mỗi (dataset, seed) ghi vào {DRIVE_DIR}/runs/kac/<ds>/<PHASE>/ ; seed đã xong được bỏ qua
# ==============================================================================
import time
def done(out): return os.path.exists(f"{out}/results-acc/global.yaml")
for ds in DATASETS:
    config = make_cfg(ds)
    for seed in SEEDS:
        base = f"{DRIVE_DIR}/runs/kac/{ds}/{PHASE}"; out, dumps = f"{base}/seed{seed}", f"{base}/dumps"
        if done(out): print(ds, PHASE, seed, "đã xong, bỏ qua"); continue
        shutil.rmtree(out, ignore_errors=True); shutil.rmtree(f"{dumps}/seed{seed}", ignore_errors=True); os.makedirs(out)
        t0 = time.time(); print(f"== {ds} | {PHASE} | seed {seed} ==", flush=True)
        r = subprocess.run(cmd(ds, seed, out, config), cwd=REPO, env=run_env(ds, seed, dumps))
        shutil.rmtree(f"{out}/models", ignore_errors=True)
        print(f"   xong sau {(time.time()-t0)/60:.1f} phút, mã thoát {r.returncode}", flush=True)
        assert r.returncode == 0, f"lỗi - xem {out}/output.log"
