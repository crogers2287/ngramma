# Local forward diagnostic harness

`ngramma_runtime` adapts the original `flash_memory` environment, artifact,
read-only table, bounded weight decoding, short-sequence forward, memory guard,
and activation-reference code. Historical `reference/` and `data/` files remain
unchanged. This package does not download weights, build or launch services,
call teachers, train, or deploy overlays. The recorded forward-parity failure
remains a closed training gate; successful imports are not numerical qualification.

Install the source with `python -m pip install -e '.[research,test]'`. The base
package has no numerical dependencies, so configuration and artifact helpers
can also be installed alone. NumPy and PyTorch version ranges describe supported
installation candidates, not a reproduced historical dependency lock. Record
exact installed versions, source revisions, and native-library hashes for each run.

Supply matching local dependencies before importing any model classes:

```sh
export NGRAMMA_ENGINE_SOURCE=/path/to/reconstructed/llama.cpp
export NGRAMMA_ENGRAFT_SOURCE=/path/to/vendor
export NGRAMMA_RUNTIME=/path/to/local/runtime
```

The engine path must contain `gguf-py/gguf/__init__.py`; an unrelated pip GGUF
reader is deliberately refused. ENGRAFT may be a parent containing `engraft/`,
the package directory itself, or (if the variable is omitted) an independently
installed package. Use the pinned ENGRAFT revision and local patch described in
`REPRODUCIBILITY.md`. Configuration checks existence and import origins; it does
**not** authenticate source revisions or model hashes. Independently verify those
against the evidence manifest before admitting any model comparison.

`NGRAMMA_RUNTIME` contains `libflash-memory-dequant.so` built against the matching
engine. `NGRAMMA_DEQUANT_LIBRARY` can override its exact path. Both the dequantization
and activation-quantization entry points must come from the matching bridge.
No native binaries are distributed. The original build has not been qualified
as a clean portable build. Linux `/proc` and a CPU-capable PyTorch installation
are required for model forward probes.

```python
from pathlib import Path
from ngramma_runtime import RuntimeConfig, configure

cfg = configure(RuntimeConfig.from_env())
cfg.validate()  # checks paths, without loading model weights or shared libraries
from ngramma_runtime.table import ModelTable
from ngramma_runtime.weights import EngineWeights
from ngramma_runtime.sequence import SequenceReplica
from ngramma_runtime.activation_reference import ActivationReference
from engraft.replica.hparams import Hparams
import torch

# Explicit matching shard lists; never paths inferred from the archived manifest.
model_paths = [Path('/path/to/model-shard.gguf')]
table_paths = [Path('/path/to/table.gguf')]
table = ModelTable(table_paths)
try:
    weights = EngineWeights(model_paths, ram_cache_bytes=1 << 30)
    hp = Hparams.from_gguf(weights.readers[0])
    replica = SequenceReplica(hp, weights, table, max_tokens=32)
    with torch.no_grad():
        capture = {}
        logits = replica.full([1, 2, 3], capture=capture)
finally:
    table.close()
```

Replace example token IDs with independently verified tokenizer outputs. Check
the pinned `Hparams` API when constructing model hyperparameters. Each full
forward starts fresh state; the sequence must fit the checkpoint's sparse
attention top-k limit. `capture` holds PLE input, layer outputs, and routing;
retaining every tensor increases RAM. `routing_source` may explicitly impose
saved routes for diagnosis and must be reported as such. `rows` is an optional
object exposing `gather(tokens)` for forward perturbation probes; no optimizer
or shared trainable row implementation is included.

`SequenceReplica.full()` requires `torch.no_grad()` and rejects activation
checkpointing. `ActivationReference(weights)` is an optional context inside
`torch.no_grad()` for investigating CPU activation rounding. It is not a gradient
estimator and does not imply parity has passed. The inherited ENGRAFT replica
methods are upstream interfaces, not an admitted training surface.

Memory guards run once per layer, defaulting to 64 GiB maximum RSS and 24 GiB
available host RAM reserve. Set `NGRAMMA_MAX_RSS_GIB` and
`NGRAMMA_MIN_AVAILABLE_GIB` explicitly for a diagnostic host. These checks are
not hard OS allocation limits and cannot prevent allocations between checks.
Never infer that a complete model fits from sparse row size.

Run lightweight validation with `python -m pytest tests/test_runtime_config.py`.
These tests use no checkpoint, GPU, or real bridge. Base module imports do not
modify `sys.path` or load NumPy, Torch, GGUF, or ENGRAFT; requesting model classes
loads the optional dependencies. Configure once before those imports; use a
fresh process for a different engine or ENGRAFT source.

Original integration code is covered by the repository MIT license. ENGRAFT
interfaces derive from Copyright 2026 fulvian, Apache-2.0; see `NOTICE`,
`licenses/ENGRAFT-Apache-2.0.txt`, and `licenses/ENGRAFT-NOTICE.txt`. ENGRAFT itself
is supplied separately. Engine licenses remain in `licenses/llama-cpp-MIT.txt`.
