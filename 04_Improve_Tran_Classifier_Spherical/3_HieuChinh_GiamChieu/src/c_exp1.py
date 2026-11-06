# ==============================================================================
# 4. THỬ NGHIỆM 1: HIỆU CHỈNH LOGIT KAC THEO KHỐI TASK
#    KAC học bằng CE trong từng task -> mỗi khối logit có độ lệch/thang riêng -> argmax trên mọi lớp hay chọn sai task.
#    Các cách sau chỉ biến đổi logit của CHÍNH ảnh đó (không học gì, không dùng nhãn, không thêm nhánh):
#      raw           : argmax như KAC gốc
#      center        : trừ trung bình logit trong từng khối         (bỏ độ lệch của khối)
#      center_half   : trừ 0.5 x trung bình khối                     (bỏ một nửa)
#      zscore        : (logit - trung bình khối) / độ lệch chuẩn khối (bỏ cả lệch lẫn thang)
#      block_softmax : log-softmax trong từng khối                   (chọn khối tự tin nhất)
# ==============================================================================
def calibrate(k, bounds, how):
    if how == "raw": return k
    out = torch.empty_like(k)
    for lo, hi in bounds:
        s = k[:, lo:hi]
        if how == "center":        out[:, lo:hi] = s - s.mean(1, keepdim=True)
        elif how == "center_half": out[:, lo:hi] = s - 0.5 * s.mean(1, keepdim=True)
        elif how == "zscore":      out[:, lo:hi] = (s - s.mean(1, keepdim=True)) / (s.std(1, keepdim=True) + 1e-6)
        elif how == "block_softmax": out[:, lo:hi] = F.log_softmax(s, 1)
        else: raise ValueError(how)
    return out

ROWS1, ROWS1F = [], []
t_all = time.time()
for (ds, seed), (d, T) in RUNS.items():
    bounds = bounds_of(d, T)
    acc = {h: np.zeros((T, T)) for h in CALIB}; tid = {h: np.zeros((T, T)) for h in CALIB}
    for t in range(T):
        bt = bounds[:t + 1]
        for j in range(t + 1):
            v = torch.load(f"{d}/test_{t}_{j}.pt"); k, y = v["k"].float(), v["y"].long(); lo, hi = bounds[j]
            for h in CALIB:
                p = calibrate(k, bt, h).argmax(1)
                acc[h][j, t] = 100.0 * (p == y).float().mean().item()
                tid[h][j, t] = 100.0 * ((p >= lo) & (p < hi)).float().mean().item()
    for h in CALIB:
        avg, last, forg = _metrics(acc[h], T)
        ROWS1.append(dict(dataset=ds, seed=seed, calib=h, avg=avg, last=last, forgetting=forg,
                          task_id_last=float(np.mean([tid[h][j, T - 1] for j in range(T)]))))
    # ghép KAC đã hiệu chỉnh với Spherical-ViT đóng băng (fuse / hier), cùng lưới tham số như trước
    for h in CALIB_FUSE:
        w = f"{OUT}/work/calib/{ds}/seed{seed}/{h}"
        link_dir(w, d, [f"train_frozen_{t}.pt" for t in range(T)] + [f"test_frozen_{j}.pt" for j in range(T)])
        for t in range(T):
            for j in range(t + 1):
                v = torch.load(f"{d}/test_{t}_{j}.pt")
                torch.save({"k": calibrate(v["k"].float(), bounds[:t + 1], h), "y": v["y"]}, f"{w}/test_{t}_{j}.pt")
        r, _ = analyse_seed_src(w, T, seed, "frozen")
        for x in r: x.update(dataset=ds, kac=h)
        ROWS1F += [x for x in r if x["variant"] in ("fuse", "hier")]
    print(f"{ds:13} seed {seed}: xong ({time.time() - t_all:.0f}s)", flush=True)
pd.DataFrame(ROWS1).to_csv(f"{OUT}/exp1_calib_kac_rows.csv", index=False)
pd.DataFrame(ROWS1F).to_csv(f"{OUT}/exp1_calib_fuse_rows.csv", index=False)
