import zipfile, glob
with zipfile.ZipFile(f"{OUT}/cost_profile_results.zip", "w") as z:
    for p in glob.glob(f"{OUT}/*.csv"): z.write(p, os.path.basename(p))
print("Tải file này về:", f"{OUT}/cost_profile_results.zip")
