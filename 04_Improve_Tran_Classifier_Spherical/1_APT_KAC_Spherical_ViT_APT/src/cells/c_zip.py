# ==============================================================================
# 7. GÓI KẾT QUẢ NHỎ (gửi lại): bảng tổng hợp, cấu hình chọn trên Val, log của repo
# ==============================================================================
import zipfile
zp = f"{DRIVE_DIR}/apt_spherical_results_{PHASE}.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for p in glob.glob(f"{DRIVE_DIR}/*.csv") + glob.glob(f"{DRIVE_DIR}/*.json"): z.write(p, os.path.basename(p))
    for p in glob.glob(f"{DRIVE_DIR}/runs/*/*/*/seed*/output.log"): z.write(p, os.path.relpath(p, DRIVE_DIR))
print("Đã tạo", zp)
