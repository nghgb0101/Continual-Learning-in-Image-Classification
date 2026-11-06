# ==============================================================================
# 5. BẢNG KẾT QUẢ (lưu CSV trong OUT)
# ==============================================================================
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
tr, sph, inf = pd.DataFrame(TRAIN_ROWS), pd.DataFrame(SPH_ROWS), pd.DataFrame(INF_ROWS)
tr.to_csv(f"{OUT}/cost_train.csv", index=False, float_format="%.4f")
sph.to_csv(f"{OUT}/cost_spherical_build.csv", index=False, float_format="%.4f")
inf.to_csv(f"{OUT}/cost_inference.csv", index=False, float_format="%.4f")
print("GPU:", GPU_NAME, "| batch train", BATCH_TRAIN, "| batch suy luận", BATCH_INFER, "| fp16 | ảnh 224x224")
print("\n(a) Train APT + KAC (đo trên Task 1, ngoại suy cho toàn bộ chuỗi task):")
print(tr[["dataset", "task1_images", "epoch_time_s", "train_img_per_s_real", "train_img_per_s_gpu", "peak_VRAM_train_GB",
          "trainable_total", "est_full_train_h_real", "est_full_train_h_gpu"]].round(3).to_string(index=False))
print("\n(b) Xây tâm Spherical (Task 1; ước tính cho cả chuỗi; bộ nhớ tâm khi đủ mọi lớp):")
print(sph[["dataset", "source", "feat_time_task1_s", "kmeans_time_task1_s", "est_build_all_tasks_s", "centres_MB_all_classes"]].round(3).to_string(index=False))
print("\n(c) Suy luận (GPU thuần, dữ liệu đã trên GPU):")
print(inf[["dataset", "config", "GFLOPs_per_img", "ms_per_img", "img_per_s", "peak_VRAM_infer_GB", "extra_backbone_params_M"]].round(3).to_string(index=False))
