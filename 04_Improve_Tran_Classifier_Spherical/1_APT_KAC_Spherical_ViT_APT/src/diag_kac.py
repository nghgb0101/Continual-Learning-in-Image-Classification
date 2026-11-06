def diag_kac(run_dir):
    """Chẩn đoán logit KAC từ dump. In ở bước cuối (mọi task đã học): với mỗi task thật j -> acc, acc nếu biết task id,
    điểm logit lớn nhất trung bình, và phân bố nơi các ảnh bị dự đoán vào."""
    for seed, (d, T) in sorted(seeds_complete(run_dir).items()):
        ld = lambda n: torch.load(os.path.join(d, n))
        Vs = [ld(f"test_{t}_{t}.pt")["k"].shape[1] for t in range(T)]
        bnd = [((Vs[t - 1] if t else 0), Vs[t]) for t in range(T)]
        t = T - 1
        print(f"\n=== seed {seed}, bước cuối ({T} task) ===")
        print("task | acc  | acc nếu biết task | max-logit TB | % bị đoán vào task: " + " ".join(f"{b:>3}" for b in range(T)))
        tot = np.zeros(T); n_all = 0
        for j in range(T):
            v = ld(f"test_{t}_{j}.pt"); k, y = v["k"].float(), v["y"].long(); lo, hi = bnd[j]
            pred = k.argmax(1); ptask = torch.tensor([[i for i, (a, b) in enumerate(bnd) if a <= p < b][0] for p in pred.tolist()])
            til = (k[:, lo:hi].argmax(1) + lo == y).float().mean().item() * 100
            hist = np.array([(ptask == b).float().mean().item() * 100 for b in range(T)])
            tot += hist * len(y); n_all += len(y)
            print(f"{j:>4} | {100 * (pred == y).float().mean().item():4.1f} | {til:17.1f} | {k.max(1).values.mean().item():12.2f} | " + "  ".join(f"{h:3.0f}" for h in hist))
        print("  tổng thể: % ảnh đoán vào từng task:", " ".join(f"{h / n_all:.0f}" for h in tot))
        # thang đo logit của từng khối lớp (trọng số của head): chuẩn của hàng trọng số
        k = ld(f"test_{t}_{T - 1}.pt")["k"].float()
        print("  logit TB theo khối lớp (trên ảnh task cuối):", " ".join(f"{k[:, a:b].mean().item():.1f}" for a, b in bnd))
