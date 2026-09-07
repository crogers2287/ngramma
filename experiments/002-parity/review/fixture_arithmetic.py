"""Read existing activation fixtures only; never load model weights or use GPUs."""
import argparse
import ctypes
import json
from pathlib import Path
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument('fixtures', type=Path)
p.add_argument('quantizer', type=Path)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
torch.set_num_threads(1)
meta = json.loads((a.fixtures / 'tensors.json').read_text())
def tensor(name, occurrence=0):
    m = [m for m in meta if m['name'] == name][occurrence]
    return np.ndarray(tuple(reversed(m['shape'])), dtype='<f4',
        buffer=(a.fixtures / m['file']).read_bytes(),
        strides=tuple(reversed(m['strides']))).copy().squeeze()

x = tensor('hc_init')
attention = tensor('linear_attn_out-0')
ffn = tensor('ffn_out-0')
last = tensor('l_last-0')
scales = []
reconstructed = np.empty_like(last)
for t in range(len(x)):
    design = np.column_stack((attention[t], ffn[t])).astype(np.float64)
    for c in range(x.shape[1]):
        scale = np.linalg.lstsq(design, (last[t,c].astype(np.float64)-x[t,c]), rcond=None)[0]
        scales.append(scale.tolist())
        reconstructed[t,c] = (x[t,c] + attention[t]*np.float32(scale[0])) + ffn[t]*np.float32(scale[1])

xt = torch.from_numpy(x)
square = xt.square()
mean_torch = square.mean(-1, keepdim=True)
mean_engine = square.double().mean(-1, keepdim=True).float()
result = {
    'fixture_combine_reconstruction': {
        'method': 'Fit two scalar scatter coefficients per token/stream from exact captured attention and FFN outputs; coefficients are inferred, not captured inject values.',
        'max_abs': float(np.max(np.abs(reconstructed-last))),
        'fitted_scatter_min': float(np.min(scales)),
        'fitted_scatter_max': float(np.max(scales)),
    },
    'initial_embedding_rms_reduction': {
        'mean_elements_different': int((mean_torch != mean_engine).sum()),
        'total_mean_elements': mean_torch.numel(),
        'mean_max_abs': float((mean_torch-mean_engine).abs().max()),
        'normalized_max_abs': float((xt*torch.rsqrt(mean_torch+1e-6)-xt/torch.sqrt(mean_engine+1e-6)).abs().max()),
        'epsilon': 1e-6,
        'note': 'Arithmetic comparison with stated illustrative epsilon; no claim this reproduces the checkpoint epsilon or ggml kernel exactly.',
    },
}
lib = ctypes.CDLL(str(a.quantizer))
quant = lib.flash_memory_activation_quant
quant.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
quant.restype = ctypes.c_int
def q(x):
    x = np.ascontiguousarray(x)
    y = np.empty_like(x)
    rc = quant(8, x.ctypes.data, y.ctypes.data, x.shape[-1], x.size//x.shape[-1])
    if rc < 0: raise RuntimeError(rc)
    return y
result['q8_0_actual_hc_fixture_one_ulp_probe'] = []
for occurrence in (0,1):
    mixed = tensor('hc_mixed-0', occurrence)
    base = q(mixed)
    for direction, toward in [('positive', np.float32(np.inf)), ('negative', np.float32(-np.inf))]:
        perturbed = np.nextafter(mixed, toward)
        changed = q(perturbed)
        delta = np.abs(changed-base)
        result['q8_0_actual_hc_fixture_one_ulp_probe'].append({
            'hc_occurrence': occurrence, 'direction': direction,
            'input_max_abs': float(np.max(np.abs(mixed-perturbed))),
            'rounded_max_abs': float(delta.max()),
            'rounded_elements_different': int(np.count_nonzero(delta)),
            'large_rounding_jumps_over_1e-5': int(np.count_nonzero(delta > 1e-5)),
        })
a.output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
