import nbformat as nbf, sys
L = lambda p: open(p).read()
lib = "# ==============================================================================\n# 2. THƯ VIỆN PHÂN TÍCH (y nguyên code Spherical đã dùng)\n# ==============================================================================\n" \
      + L("/home/claude/apt_kac_merge/fusion_online.py") + "\n\n" + L("/home/claude/apt_kac_merge/fusion_apt.py")
md0 = """# Thử nghiệm trên dump có sẵn: hiệu chỉnh logit KAC và giảm chiều Spherical (4 dataset, Val, 3 seed)

Không train, không cần GPU hay Internet: chỉ đọc dump của các phiên APT đã chạy (logit KAC + đặc trưng ViT đóng băng).

**Thử nghiệm 1 — hiệu chỉnh logit KAC theo khối task.** KAC học bằng CE trong từng task nên mỗi khối logit có độ lệch riêng, argmax trên mọi lớp hay chọn sai task. Thử các phép biến đổi không cần học (`center`, `center_half`, `zscore`, `block_softmax`): 0 tham số, 0 nhánh phụ, vẫn 1 lượt forward. Sau đó ghép KAC đã hiệu chỉnh với Spherical-ViT (fuse/hier) xem còn tăng không.

**Thử nghiệm 2 — giảm chiều đặc trưng Spherical** 768 → d ∈ {64, 128, 256} bằng PCA (học từ ảnh train task 1 rồi đóng băng) hoặc chiếu ngẫu nhiên trực giao. So với bản đầy đủ 768 chiều trên cùng seed.

**Kaggle:** Accelerator **None** (CPU, không tốn quota GPU). Add Input output các notebook APT chứa `runs/kac/<dataset>/val/dumps/seed*` (giống notebook Swin; không cần dataset ảnh). Run All.
"""
cells = [nbf.v4.new_markdown_cell(md0),
         nbf.v4.new_markdown_cell("## 1. Cấu hình"), nbf.v4.new_code_cell(L("c_config.py")),
         nbf.v4.new_markdown_cell("## 2. Thư viện phân tích"), nbf.v4.new_code_cell(lib),
         nbf.v4.new_markdown_cell("## 3. Tìm dump"), nbf.v4.new_code_cell(L("c_find.py")),
         nbf.v4.new_markdown_cell("## 4. Thử nghiệm 1: hiệu chỉnh logit KAC"), nbf.v4.new_code_cell(L("c_exp1.py")),
         nbf.v4.new_markdown_cell("## 5. Thử nghiệm 2: giảm chiều Spherical"), nbf.v4.new_code_cell(L("c_exp2.py")),
         nbf.v4.new_markdown_cell("## 6. Bảng kết quả"), nbf.v4.new_code_cell(L("c_report.py")),
         nbf.v4.new_code_cell(L("c_zip.py"))]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"}})
nbf.write(nb, sys.argv[1]); print("wrote", sys.argv[1])
