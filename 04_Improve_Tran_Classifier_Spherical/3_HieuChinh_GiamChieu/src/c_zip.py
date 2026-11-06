import zipfile
z = f"/kaggle/working/exp_calib_proj_{PHASE}.zip" if OUT.startswith("/kaggle") else f"{OUT}/results_{PHASE}.zip"
with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
    for p in glob.glob(f"{OUT}/*.csv"): f.write(p, os.path.basename(p))
shutil.rmtree(f"{OUT}/work", ignore_errors=True)       # thư mục tạm (symlink + logit đã hiệu chỉnh), không cần giữ
print("Tải về:", z)
