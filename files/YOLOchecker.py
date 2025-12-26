#!/usr/bin/env python3
import os, sys, json, re, platform, subprocess, ctypes, glob

# ---------- helpers ----------
def run(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True, text=True, timeout=5)
        return out.strip()
    except Exception:
        return None

def read_file(p):
    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return None

def find_cuda_version_from_path(path):
    if not path:
        return None
    candidates = [
        os.path.join(path, "version.json"),
        os.path.join(path, "version.txt"),
        os.path.join(path, "lib", "version.json"),
        os.path.join(path, "lib64", "version.json"),
    ]
    for p in candidates:
        txt = read_file(p)
        if not txt:
            continue
        try:
            data = json.loads(txt)
            v = data.get("version") or data.get("cuda", {}).get("version")
            if v:
                return v
        except Exception:
            m = re.search(r"CUDA\s*Version\s*([\d.]+)", txt, re.I)
            if m:
                return m.group(1)
    base = os.path.basename(path).lower()
    m = re.search(r"v(\d+\.\d+(?:\.\d+)?)", base)
    if m:
        return m.group(1)
    return None

def parse_nvcc_version(s):
    if not s:
        return None
    m = re.search(r"release\s+([\d.]+)", s)
    return m.group(1) if m else None

def _cudnn_version_from_libpath(libpath):
    try:
        lib = ctypes.CDLL(libpath)
        func = lib.cudnnGetVersion
        func.restype = ctypes.c_size_t
        ver_num = func()
        major = ver_num // 1000
        minor = (ver_num % 1000) // 100
        patch = ver_num % 100
        return f"{major}.{minor}.{patch}"
    except Exception:
        # fallback: try to infer from filename like cudnn64_9.dll or cudnn64_8.dll
        name = os.path.basename(libpath).lower()
        m = re.search(r"cudnn(?:64_)?(\d+)\.dll", name)
        if m:
            return f"{m.group(1)}.x"
    return None

def cudnn_version_from_ctypes_names():
    names = [
        "cudnn64_9.dll", "cudnn64_8.dll", "cudnn64_7.dll",        # Windows
        "libcudnn.so.9", "libcudnn.so.8", "libcudnn.so.7",        # Linux
        "libcudnn.9.dylib", "libcudnn.8.dylib", "libcudnn.dylib", # macOS
        "cudnn"
    ]
    for n in names:
        v = _cudnn_version_from_libpath(n)
        if v:
            return v, n
    return None, None

def cudnn_version_from_windows_paths():
    if os.name != "nt":
        return None, None
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    pfx86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    bases = [pf, pfx86]
    subroots = [
        "",  # full scan under base
        os.path.join("NVIDIA GPU Computing Toolkit", "CUDA"),
        os.path.join("NVIDIA Corporation"),
        os.path.join("NVIDIA", "CUDNN"),
        os.path.join("NVIDIA Corporation", "CUDNN"),
    ]
    patterns = ["cudnn64_*.dll", "cudnn*.dll"]
    candidates = []
    for b in bases:
        for s in subroots:
            root = os.path.join(b, s) if s else b
            if not os.path.isdir(root):
                continue
            for pat in patterns:
                try:
                    candidates += glob.glob(os.path.join(root, "**", pat), recursive=True)
                except Exception:
                    pass
    # de-dup and prefer newest
    seen = set()
    uniq = []
    for p in candidates:
        p = os.path.abspath(p)
        if p.lower() in seen:
            continue
        seen.add(p.lower())
        uniq.append(p)
    uniq.sort(key=lambda x: (-os.path.getmtime(x), len(x)))
    for path in uniq:
        v = _cudnn_version_from_libpath(path)
        if v:
            return v, path
    return None, None

# ---------- checks ----------
report = []

# System
report.append({
    "component": "python",
    "present": True,
    "version": platform.python_version(),
    "details": sys.executable
})

# torch
torch_info = {"component": "torch", "present": False, "version": None, "details": None}
try:
    import torch
    torch_info["present"] = True
    torch_info["version"] = getattr(torch, "__version__", None)
    cuda_compiled = getattr(torch.version, "cuda", None)
    cuda_runtime_ok = torch.cuda.is_available()
    cudnn_ok = torch.backends.cudnn.is_available()
    cudnn_ver = None
    try:
        cudnn_ver = torch.backends.cudnn.version()
        if isinstance(cudnn_ver, int):
            major = cudnn_ver // 1000
            minor = (cudnn_ver % 1000) // 100
            patch = cudnn_ver % 100
            cudnn_ver = f"{major}.{minor}.{patch}"
    except Exception:
        pass
    gpu = None
    if cuda_runtime_ok:
        try:
            gpu = torch.cuda.get_device_name(0)
        except Exception:
            gpu = "cuda available"
    torch_info["details"] = f"cuda_compiled={cuda_compiled}, cuda_available={cuda_runtime_ok}, cudnn_available={cudnn_ok}, cudnn={cudnn_ver}, gpu={gpu}"
except Exception as e:
    torch_info["details"] = f"import failed: {e}"
report.append(torch_info)

# cuDNN
cudnn_info = {"component": "cudnn", "present": False, "version": None, "details": None}

# 1) prefer torch
tn = None
try:
    if torch_info["present"]:
        m = re.search(r"cudnn=([0-9.]+)", torch_info["details"] or "")
        if m:
            tn = m.group(1)
except Exception:
    pass
if tn:
    cudnn_info.update({"present": True, "version": tn, "details": "via torch.backends.cudnn"})
else:
    # 2) try well-known names
    v, src = cudnn_version_from_ctypes_names()
    if v:
        cudnn_info.update({"present": True, "version": v, "details": f"via ctypes: {src}"})
    else:
        # 3) scan Windows Program Files and Program Files (x86)
        v2, path2 = cudnn_version_from_windows_paths()
        if v2:
            cudnn_info.update({"present": True, "version": v2, "details": f"via path: {path2}"})
        else:
            cudnn_info["details"] = "not found in PATH/LD_LIBRARY_PATH or Windows Program Files"
report.append(cudnn_info)

# CUDA toolkit
cuda_info = {"component": "cudatoolkit", "present": False, "version": None, "details": None}
nvcc_out = run("nvcc --version")
nvcc_ver = parse_nvcc_version(nvcc_out) if nvcc_out else None
cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME") or (os.path.exists("/usr/local/cuda") and "/usr/local/cuda") or None
cuda_path_ver = find_cuda_version_from_path(cuda_path) if cuda_path else None

torch_cuda_compiled = None
try:
    if torch_info["present"]:
        torch_cuda_compiled = getattr(torch.version, "cuda", None)
except Exception:
    pass

if nvcc_ver or cuda_path_ver or torch_cuda_compiled:
    cuda_info["present"] = True
    cuda_info["version"] = nvcc_ver or cuda_path_ver or torch_cuda_compiled
    detail_parts = []
    if nvcc_ver: detail_parts.append("via nvcc")
    if cuda_path and cuda_path_ver: detail_parts.append(f"via {cuda_path}")
    if torch_cuda_compiled and not (nvcc_ver or cuda_path_ver): detail_parts.append("torch compiled against this CUDA")
    cuda_info["details"] = ", ".join(detail_parts) if detail_parts else None
else:
    cuda_info["details"] = "nvcc not found, no CUDA_PATH, torch not CUDA-enabled"
report.append(cuda_info)

# opencv-python
cv_info = {"component": "cv2", "present": False, "version": None, "details": None}
try:
    import cv2
    cv_info["present"] = True
    cv_info["version"] = getattr(cv2, "__version__", None)
    try:
        build = cv2.getBuildInformation()
        cuda_yes = bool(re.search(r"CUDA:\s+YES", build))
        cv_info["details"] = f"cuda_build={cuda_yes}"
    except Exception:
        pass
except Exception as e:
    cv_info["details"] = f"import failed: {e}"
report.append(cv_info)

# ultralytics
ultra_info = {"component": "ultralytics", "present": False, "version": None, "details": None}
try:
    import ultralytics
    ultra_info["present"] = True
    ver = getattr(ultralytics, "__version__", None)
    if not ver:
        try:
            from ultralytics import __version__ as _v
            ver = _v
        except Exception:
            pass
    ultra_info["version"] = ver
except Exception as e:
    ultra_info["details"] = f"import failed: {e}"
report.append(ultra_info)

# nvidia-smi (optional)
smi = run("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader")
if smi:
    report.append({"component": "nvidia-smi", "present": True, "version": None, "details": smi})

# ---------- output ----------
widths = {
    "component": max(len("component"), max(len(r["component"]) for r in report)),
    "present": len("present"),
    "version": max(len("version"), max(len(str(r["version"])) if r["version"] else 1 for r in report)),
}
header = f"{'component'.ljust(widths['component'])}  {'present':7}  {'version'.ljust(widths['version'])}  details"
print(header)
print("-" * len(header))
missing = []
for r in report:
    present = "yes" if r["present"] else "no"
    version = r["version"] or "-"
    details = r.get("details") or ""
    print(f"{r['component'].ljust(widths['component'])}  {present:7}  {str(version).ljust(widths['version'])}  {details}")
    if r["component"] in ("torch", "cv2", "ultralytics", "cudatoolkit", "cudnn") and not r["present"]:
        missing.append(r["component"])

print("\nmissing:", ", ".join(missing) if missing else "none")