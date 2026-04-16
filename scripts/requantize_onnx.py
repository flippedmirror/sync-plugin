"""Requantize ONNX models skipping Conv ops (which lack ARM/macOS support).

Only quantize MatMul and attention layers — these are the bulk of computation.
"""

import os
from onnxruntime.quantization import QuantType, quantize_dynamic

MODEL_DIR = "models/florence2-onnx"

for name in ["encoder", "decoder"]:
    src = os.path.join(MODEL_DIR, f"{name}.onnx")
    dst = os.path.join(MODEL_DIR, f"{name}_int8.onnx")

    print(f"Quantizing {name}...")
    quantize_dynamic(
        model_input=src,
        model_output=dst,
        weight_type=QuantType.QInt8,
        nodes_to_exclude=[],
        # Only quantize these op types (skip Conv which has no ARM int8 impl)
        op_types_to_quantize=["MatMul", "Gemm", "Attention"],
    )
    orig = os.path.getsize(src) / 1024 / 1024
    quant = os.path.getsize(dst) / 1024 / 1024
    print(f"  {orig:.1f} MB → {quant:.1f} MB ({quant/orig:.0%})")
