import importlib

MODULES = [
    "numpy",
    "torch",
    "hydra",
    "omegaconf",
    "cv2",
    "zarr",
    "numcodecs",
    "wandb",
    "diffusers",
    "accelerate",
    "av",
    "mujoco",
    "glfw",
    "mink",
]

failed = []
for name in MODULES:
    try:
        importlib.import_module(name)
    except Exception as e:  # pragma: no cover
        failed.append((name, str(e)))

print("=== Environment Verification ===")
if failed:
    for name, err in failed:
        print(f"[FAIL] {name}: {err}")
    raise SystemExit(1)

import torch
print("[OK] all modules imported")
print(f"[INFO] torch={torch.__version__}, cuda_available={torch.cuda.is_available()}")
