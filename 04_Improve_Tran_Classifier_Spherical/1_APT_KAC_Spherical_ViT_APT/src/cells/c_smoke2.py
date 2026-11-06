# ==============================================================================
# 4. CHẠY THỬ NHANH (~10 phút): DATASETS[0], seed SEEDS[0], 2 task × 2 epoch
# ==============================================================================
SMOKE_DS = DATASETS[0]
shutil.rmtree("/tmp/smoke", ignore_errors=True)
r = subprocess.run(cmd(SMOKE_DS, SEEDS[0], "/tmp/smoke/out", make_cfg(SMOKE_DS, max_task=2, tag="smoke"), epochs=2),
                   cwd=REPO, env=run_env(SMOKE_DS, SEEDS[0], "/tmp/smoke/dumps"), capture_output=True, text=True)
print(r.stdout[-1500:]); print(r.stderr[-1500:] if r.returncode else "")
assert r.returncode == 0, "chạy thử lỗi - gửi mình đoạn lỗi ở trên"
assert "Shuffling" not in r.stdout, "repo vẫn tự xáo lớp (rand_split) - thứ tự task sẽ sai"
rows, diag = analyse_all("/tmp/smoke/dumps")
import pandas as pd
print(pd.DataFrame(rows).groupby(["source", "variant"])[["avg", "last"]].max().round(1)); print(pd.DataFrame(diag))
print("Chạy thử OK (số liệu 2 epoch chưa có ý nghĩa)")
