import hashlib, pathlib
A, B = pathlib.Path("PROJECT-V1/fedgkt"), pathlib.Path("PROJECT-V1/fedgkt_v2")
files = ["src/models/gat.py", "src/models/fedgkt.py", "src/data/pkg.py",
         "src/data/pkg_batched.py", "src/training/centralised_batched.py",
         "src/utils/config.py", "src/utils/metrics.py",
         "data/processed/edge_index.pt", "data/processed/pkgs/pkg_233536.pt",
         "data/processed/pkgs/pkg_45224.pt", "data/processed/pkgs/pkg_21419.pt"]
h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()[:12] if p.exists() else "MISSING"
for f in files:
    a, b = h(A / f), h(B / f)
    print("SAME" if a == b != "MISSING" else "DIFF", f, a, b)