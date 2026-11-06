import zipfile
z = f"/kaggle/working/swin_spherical_results_{PHASE}.zip" if OUT.startswith("/kaggle") else f"{OUT}/results_{PHASE}.zip"
with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
    for p in glob.glob(f"{OUT}/*.csv"): f.write(p, os.path.basename(p))
print("Tải về:", z)
