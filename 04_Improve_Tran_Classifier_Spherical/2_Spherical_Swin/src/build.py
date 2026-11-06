import nbformat as nbf, sys
L = lambda p: open(p).read()
lib = L("/home/claude/apt_kac_merge/fusion_online.py") + "\n\n" + L("/home/claude/apt_kac_merge/fusion_apt.py")
lib = "# ==============================================================================\n# 4. THƯ VIỆN PHÂN TÍCH (y nguyên code đã dùng cho Spherical trên ViT / APT)\n# ==============================================================================\n" + lib
md0 = """# Spherical trên Swin đóng băng — 4 dataset (Val, 3 seed)

Thay **ViT đóng băng** bằng **Swin đóng băng** (Swin-B và Swin-T) trong nhánh Spherical; mọi thứ khác giữ nguyên (split của nhóm, spherical k-means, κ = κ_mult / median khoảng cách, cùng lưới k, κ_mult, w, λ, w_in).

| biến thể | nhánh Spherical | KAC |
|---|---|---|
| `proto` | backbone đóng băng, đứng một mình | không dùng |
| `fuse` | backbone đóng băng | logit KAC của APT-ViT (dump có sẵn) |
| `hier` | backbone đóng băng chọn task | KAC chọn lớp trong task |

Bảng in ra đặt ViT, Swin-B, Swin-T cạnh nhau trên cùng seed, cùng ảnh, cùng logit KAC, kèm chi phí (tham số, GFLOPs, ảnh/giây). Không train gì: mỗi ảnh chỉ đi qua mỗi backbone một lần.

**Kaggle:** Settings → GPU + Internet On. Add Input:
1. dataset CIFAR-100, CUB-200, Stanford Cars, ImageNet-R của nhóm;
2. output các notebook APT đã chạy (chứa `runs/kac/<dataset>/val/dumps/seed*`) cho đủ seed 0, 1, 2 của từng dataset. Một notebook có nhiều version: version cũ → tab Output → New dataset, rồi Add Input dataset đó.

Seed nào không có dump KAC thì chỉ tính được `proto` (notebook tự nhận ra, in "KHÔNG CÓ" trong log).
"""
cells = [nbf.v4.new_markdown_cell(md0),
         nbf.v4.new_markdown_cell("## 1. Cấu hình"), nbf.v4.new_code_cell(L("c_config.py")),
         nbf.v4.new_markdown_cell("## 2. Dữ liệu và chia seed theo quy trình của nhóm"), nbf.v4.new_code_cell(L("c_data.py")),
         nbf.v4.new_markdown_cell("## 3. Trích đặc trưng Swin đóng băng"), nbf.v4.new_code_cell(L("c_feat.py")),
         nbf.v4.new_markdown_cell("## 4. Thư viện phân tích"), nbf.v4.new_code_cell(lib),
         nbf.v4.new_markdown_cell("## 5. Chạy"), nbf.v4.new_code_cell(L("c_run.py")),
         nbf.v4.new_markdown_cell("## 6. Bảng kết quả"), nbf.v4.new_code_cell(L("c_report.py")),
         nbf.v4.new_code_cell(L("c_zip.py"))]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                                "language_info": {"name": "python"}})
nbf.write(nb, sys.argv[1]); print("wrote", sys.argv[1])
