import nbformat as nbf, os
D = os.path.dirname(os.path.abspath(__file__)) + "/cells/"
L = lambda f: open(D + f).read()
intro = """# Đo chi phí tính toán: APT + KAC và nhánh Spherical (4 dataset)

Notebook này **không train đầy đủ**. Với mỗi dataset, notebook train APT + KAC trên Task 1 trong vài epoch để đo tốc độ, rồi ngoại suy ra thời gian train cả chuỗi task. Sau đó đo chi phí của nhánh Spherical và chi phí suy luận của từng cấu hình.

| Phần | Đo gì |
|---|---|
| (a) Train APT + KAC | thời gian/epoch thực tế (gồm đọc ảnh), tốc độ GPU thuần (ảnh/giây), VRAM đỉnh, số tham số trainable (prompt + head KAC), ước tính giờ train cả chuỗi |
| (b) Xây tâm Spherical | thời gian trích đặc trưng + spherical k-means cho Task 1, ước tính cho cả chuỗi, bộ nhớ tâm khi đủ mọi lớp |
| (c) Suy luận | GFLOPs/ảnh, ms/ảnh, ảnh/giây, VRAM đỉnh cho: APT + KAC; + Spherical (APT stale / ViT đóng băng / Swin-B / Swin-T) |

Trọng số pretrained không ảnh hưởng thời gian, bộ nhớ hay FLOPs, nên mặc định `PRETRAINED = False` (không cần Internet nếu Kaggle đã có `timm`).
Kiến trúc giống repo APT (24 prompt cộng vào token CLS) và KAC (4 RBF); đây là bản tự cài để đo, không phải repo chính thức.

**Cách chạy:** bật GPU T4 → Add Input 4 dataset của nhóm → Run All (khoảng 15–25 phút)."""
cells = [nbf.v4.new_markdown_cell(intro),
         nbf.v4.new_markdown_cell("## 1. Cấu hình"), nbf.v4.new_code_cell(L("config.py")),
         nbf.v4.new_markdown_cell("## 2. Dữ liệu (cùng cách chia seed với các notebook của nhóm)"), nbf.v4.new_code_cell(L("data.py")),
         nbf.v4.new_markdown_cell("## 3. Mô hình và hàm đo"), nbf.v4.new_code_cell(L("model.py")),
         nbf.v4.new_markdown_cell("## 4. Đo chi phí"), nbf.v4.new_code_cell(L("run.py")),
         nbf.v4.new_markdown_cell("## 5. Bảng kết quả"), nbf.v4.new_code_cell(L("report.py")), nbf.v4.new_code_cell(L("zip.py"))]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "name": "python3"}, "accelerator": "GPU"})
nbf.write(nb, os.path.dirname(D[:-1]) + "/cost_profile_4ds.ipynb")
for i, c in enumerate(cells):
    if c.cell_type == "code": compile(c.source, f"cell{i}", "exec")
print("ok", len(cells))
