import os
import sys
import platform
import subprocess
import importlib.metadata as im

# ============================================================
# Environment flags
# ============================================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ============================================================
# Required packages
# None = any version
# ============================================================
BASE_REQUIRED = {
    "numpy":"1.26.4",
    "torch":None,
    "transformers":"4.46.3",
    "datasets":None,
    "evaluate":None,
    "scikit-learn":None,
    "imbalanced-learn":None,
    "wordcloud":None,
    "umap-learn":None,
}

INSTALL_SPECS = {
    "numpy":"numpy==1.26.4",
    "torch":"torch",                                                # Linux wheels from PyPI include CUDA
    "transformers":"transformers==4.46.3",
}


# ============================================================
# Helpers
# ============================================================
def run(cmd,check=True):
    print("\nRUN:")
    print(" ".join(cmd))

    result = subprocess.run(
        cmd,
        text=True,
    )

    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode,cmd)

    return result


def get_version(pkg):
    try:
        return im.version(pkg)
    except im.PackageNotFoundError:
        return None


def detect_base():
    base = {}

    base["python"]       = sys.version
    base["executable"]   = sys.executable
    base["platform"]     = platform.platform()
    base["conda_prefix"] = os.environ.get("CONDA_PREFIX")
    base["virtual_env"]  = os.environ.get("VIRTUAL_ENV")
    base["is_root"]      = hasattr(os,"geteuid") and os.geteuid() == 0
    return base


def print_base(base):
    print("\nEnvironment detected:")
    print("Python:",base["python"])
    print("Executable:",base["executable"])
    print("Platform:",base["platform"])
    print("Conda:",base["conda_prefix"])
    print("Virtualenv:",base["virtual_env"])
    print("Root user:",base["is_root"])


def missing_packages():
    missing = []

    for pkg,required_version in BASE_REQUIRED.items():
        installed_version = get_version(pkg)

        if installed_version is None:
            print(f"Missing: {pkg}")
            missing.append(pkg)
            continue

        if required_version is not None and installed_version != required_version:
            print(f"Version mismatch: {pkg} installed={installed_version}, required={required_version}")
            missing.append(pkg)

    return missing


def check_torch():
    import torch
    import transformers

    print("\nPyTorch check:")
    print("torch:",torch.__version__)
    print("transformers:",transformers.__version__)
    print("CUDA available:",torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:",torch.cuda.get_device_name(0))


# ============================================================
# Main installer
# ============================================================
def prepare_env(force=False):
    base = detect_base()
    print_base(base)

    missing = list(BASE_REQUIRED) if force else missing_packages()

    if len(missing) == 0:
        print("\nEnvironment packages OK. No installation needed.")
        check_torch()
        return

    print("\nInstalling required packages...")

    specs = [INSTALL_SPECS.get(pkg,pkg) for pkg in missing]

    if "transformers" in missing:
        specs.append("tokenizers<0.21")                             # matches transformers 4.46.3

    run(
        [
            sys.executable,"-m","pip","install",
            "--no-cache-dir",
            *specs,
        ],
        check=True,
    )

    print("\nPip check:")
    run([sys.executable,"-m","pip","check"],check=False)

    print("\nDone. Restart the kernel/runtime before importing torch or transformers.")
    print("After restart, run:")
    print("prepare_env(force=False)")
