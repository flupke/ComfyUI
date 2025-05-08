# ruff: noqa: T201
import sys
from safetensors import safe_open

# Replace with the actual path to your model file
model_path = sys.argv[1]
tensor_dtypes = {}

with safe_open(model_path, framework="pt", device="cpu") as f:
    for key in f.keys():
        tensor = f.get_tensor(key)
        if tensor.dtype not in tensor_dtypes:
            tensor_dtypes[tensor.dtype] = 0
        tensor_dtypes[tensor.dtype] += 1
        # To see every tensor's dtype:

print("\nSummary of tensor dtypes found in the file:")
for dtype, count in tensor_dtypes.items():
    print(f"- {dtype}: {count} tensor(s)")
