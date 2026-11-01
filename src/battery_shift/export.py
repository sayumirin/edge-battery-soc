"""CPU dynamic int8 GRU/Linear plus independently verified float32 ONNX."""

import copy

import numpy as np
import onnxruntime as ort
import torch
from torch import nn


def export_models(model, sample, out):
    model = copy.deepcopy(model).cpu().eval()
    torch.save(model.state_dict(), out / "float32.pt")
    quantized = torch.ao.quantization.quantize_dynamic(model, {nn.GRU, nn.Linear}, dtype=torch.qint8)
    torch.save(quantized.state_dict(), out / "int8.pt")
    torch.onnx.export(
        model,
        sample,
        out / "model.onnx",
        input_names=["x"],
        output_names=["soc"],
        dynamic_axes={"x": {0: "batch", 1: "time"}, "soc": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    session = ort.InferenceSession(str(out / "model.onnx"), providers=["CPUExecutionProvider"])
    error = 0.0
    with torch.inference_mode():
        for x in [sample, sample[:1, : max(1, sample.shape[1] // 2)]]:
            p = session.run(None, {"x": x.numpy()})[0]
            error = max(error, float(np.max(abs(p - model(x).numpy()))))
    if error > 1e-5:
        raise ValueError(f"ONNX parity failure: {error}")
    return quantized, dict(
        float32_bytes=(out / "float32.pt").stat().st_size,
        int8_bytes=(out / "int8.pt").stat().st_size,
        onnx_bytes=(out / "model.onnx").stat().st_size,
        onnx_max_abs_error=error,
    )
