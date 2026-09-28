"""Export deep-sort-realtime's MobileNetV2 appearance embedder to ONNX for `async_yolo_test.py --embedder onnx`.

Writes tools/models/deepsort_mobilenetv2_embedder.onnx (~9 MB, not committed: re-run this script instead).
The weights are the ones that ship inside the deep-sort-realtime package, so the ONNX model gives the same
features as the torch embedder DeepSort(...) uses by default. Input 'x' is the library's own preprocessing
(N x 3 x 224 x 224, ImageNet-normalised RGB); output 'f' is the N x 1280 feature.

Usage (from the repo root):  python tools/export_embedder_onnx.py
Needs: torch (already in the project env), onnx for the export, onnxruntime for the parity check at the end.
"""
import os
import sys
import numpy as np
import torch
try:
    import onnx  # noqa: F401  torch.onnx.export needs it
except ImportError:
    sys.exit("The export needs the 'onnx' package: pip install onnx onnxruntime")
from deep_sort_realtime.embedder.embedder_pytorch import MobileNetv2_Embedder

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "models", "deepsort_mobilenetv2_embedder.onnx")

emb = MobileNetv2_Embedder(half=False, max_batch_size=16, bgr=True, gpu=False)  # CPU, fp32 (same as DeepSort on CPU)
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
export_kwargs = dict(input_names=["x"], output_names=["f"], dynamic_axes={"x": {0: "n"}, "f": {0: "n"}}, opset_version=17)
try:
    # torch >= 2.9 defaults to the new dynamo exporter; the classic exporter is the one that was tested
    torch.onnx.export(emb.model, torch.zeros(1, 3, 224, 224), OUT_PATH, dynamo=False, **export_kwargs)
except TypeError:  # older torch without the dynamo argument
    torch.onnx.export(emb.model, torch.zeros(1, 3, 224, 224), OUT_PATH, **export_kwargs)
print(f"wrote {OUT_PATH} ({os.path.getsize(OUT_PATH) / 1048576:.1f} MB)")

# Parity check: random "crops" of different sizes through the library's torch path vs ONNX Runtime
try:
    import onnxruntime as ort
except ImportError:
    print("onnxruntime not installed, skipping the parity check (pip install onnxruntime)")
else:
    rng = np.random.default_rng(0)
    crops = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for h, w in [(40, 70), (120, 200), (300, 180), (64, 64)]]
    ref = np.stack(emb.predict(crops))
    x = torch.cat([emb.preprocess(c) for c in crops]).numpy()
    sess = ort.InferenceSession(OUT_PATH, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"x": x})[0]
    cos = (out * ref).sum(1) / np.linalg.norm(out, axis=1) / np.linalg.norm(ref, axis=1)
    print(f"torch vs onnxruntime: max abs diff {np.abs(out - ref).max():.2e}, min cosine similarity {cos.min():.6f}")
