# Trace memory rows without loading Flash Next

From a source checkout, run:

```sh
PYTHONPATH=src python3 -m ngramma_runtime.addressing \
  --identity data/model-identity.json \
  --batch experiments/005-behavior/discovery.json \
  --job dev-decimal-0 --row 170852069
```

This reports that global row `170852069` occurs at token position `38`, head
`8`, in the saved prompt. It needs Python 3.10+ and no model files, NumPy,
Torch, GPU, API key, or original absolute paths. After installing the wheel,
the same command is available as `ngramma-addresses`.

Omit `--row` to print all sixteen global addresses at every position. Repeat
`--row` to inspect several rows. Add `--include-decoded` to include the generated
tokens that were actually fed back into the native generation loop. Its final
sample is excluded because the loop stops before feeding that token back.
Alternatively, use `--tokens tokens.json` with a JSON array of token IDs.

## Use from Python

```python
import json
from ngramma_runtime.addressing import Addressing

identity = json.load(open("data/model-identity.json"))
addressing = Addressing.from_metadata(identity["architecture_metadata"])
rows = addressing.addresses([15666, 1132, 357])  # Saved IDs for "Answer only A"
print(rows[-1][8:])  # Eight trigram-head global addresses
```

The final triple produces the same eight target addresses as it does inside
the complete prompt. Earlier positions also include sequence-boundary padding,
so their rows are different. Supply complete token history for a continuing
sequence and call separately for each fresh sequence.

## What is checked

The implementation matches all 624 row IDs in the historical native engine's
39-token reference trace, covering ordinary text, Unicode/chat, repeated
tokens, and EOS boundaries. It also reproduces all 24 instruction-row usage
records and all eight sets of reminder-donor rows in experiment 006. Tests cover
unsigned 64-bit wraparound, invalid geometry, overlapping head ranges, and
dependency-free CLI execution.

The PLE EOS token in this checkpoint is `248044`. A predecessor equal to that
token cuts older hash history; a current EOS does not discard its own
predecessors. The chat delimiter `248046` has a different ID and does not
automatically trigger that rule. These values come from model metadata, not
from treating every end-of-generation token as the memory EOS.

## Limits

This calculates addresses for one contiguous text-token sequence. It does not
tokenize text, validate IDs against a tokenizer vocabulary, authenticate model
files, decode memory vectors, or simulate packed/missing KV cells or image
embedding batches. Only the tested qwen4exp trigram/eight-head geometry is
accepted. The redacted public identity works because this tool reads metadata
only; it remains unsuitable for loading an overlay into a model.

An observed token triple is not the row's unique meaning. Hash collisions can
make other triples use that row. The tool helps locate an edit's possible
reach; generated-answer tests must establish whether it helps.
