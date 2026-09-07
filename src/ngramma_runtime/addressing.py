"""Weight-free qwen4exp row addressing for one contiguous token sequence.

Derived from ENGRAFT's addressing and the archived llama.cpp implementation;
see NOTICE and the retained Apache-2.0/MIT notices. This implements integer
address arithmetic, not tokenization, model authentication, or inference.
"""
from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path


def integer(value, lower, upper, name):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f'{name} must be an integer in [{lower}, {upper}]')
    return value


@dataclass(frozen=True)
class Addressing:
    eos_token_id: int
    multipliers: tuple[int, ...]
    head_sizes: tuple[int, ...]
    head_offsets: tuple[int, ...]

    def __post_init__(self):
        for name in ('multipliers', 'head_sizes', 'head_offsets'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        integer(self.eos_token_id, 0, 2**31 - 1, 'PLE EOS token')
        if len(self.multipliers) != 3 or len(self.head_sizes) != 16 or len(self.head_offsets) != 16:
            raise ValueError('Only the qualified 3-gram / 8-head-per-order geometry is supported')
        for x in self.multipliers:
            integer(x, 0, 2**64 - 1, 'hash multiplier')
        previous_end = 0
        for offset, size in zip(self.head_offsets, self.head_sizes):
            integer(offset, 0, 2**31 - 1, 'head offset')
            integer(size, 1, 2**31, 'head size')
            if offset < previous_end or offset + size > 2**31:
                raise ValueError('Head ranges overlap or exceed int32 row IDs')
            previous_end = offset + size

    @classmethod
    def from_metadata(cls, metadata):
        prefix = 'qwen4exp.ple.'
        order = integer(metadata.get(prefix + 'ngram_size'), 2, 8, 'ngram size')
        heads = integer(metadata.get(prefix + 'heads_per_ngram'), 1, 32, 'heads per order')
        if order != 3 or heads != 8:
            raise ValueError('Require explicit qwen4exp 3-gram / 8-head-per-order metadata')
        return cls(metadata[prefix + 'eos_token_id'],
                   tuple(metadata[prefix + 'layer_multipliers']),
                   tuple(metadata[prefix + 'head_vocab_sizes']),
                   tuple(metadata[prefix + 'head_offsets']))

    def addresses(self, tokens):
        """Global joined-table rows for each actual input token, head order 0..15.

        Starts a fresh sequence. Supply complete history to continue decoding.
        A predecessor equal to the PLE EOS token cuts older history; a current
        EOS token retains its own predecessors. Other chat delimiters do not
        automatically reset history. Packed KV gaps and image-embedding batches
        are outside this text-token interface.
        """
        tokens = list(tokens)
        for token in tokens:
            integer(token, 0, 2**31 - 1, 'token ID')
        result = []
        mask = 2**64 - 1
        for position, token in enumerate(tokens):
            context, cut = [token], False
            for distance in (1, 2):
                previous = tokens[position-distance] if position >= distance and not cut else self.eos_token_id
                cut = cut or previous == self.eos_token_id
                context.append(self.eos_token_id if cut else previous)
            rows = []
            mixed = (context[0] * self.multipliers[0]) & mask
            for order in (2, 3):
                mixed ^= (context[order-1] * self.multipliers[order-1]) & mask
                for head in range((order-2)*8, (order-1)*8):
                    rows.append(self.head_offsets[head] + mixed % self.head_sizes[head])
            result.append(rows)
        return result

    def occurrences(self, tokens, row_ids):
        """Observed uses only; a hash address can also be used by other n-grams."""
        wanted = set(row_ids)
        for row in wanted:
            integer(row, 0, 2**31 - 1, 'row ID')
            if not any(offset <= row < offset + size for offset, size in zip(self.head_offsets, self.head_sizes)):
                raise ValueError('Selected row is outside declared head ranges')
        hits = {row: [] for row in sorted(wanted)}
        for position, rows in enumerate(self.addresses(tokens)):
            for head, row in enumerate(rows):
                if row in hits:
                    hits[row].append([position, head])
        return [{'row_id': row, 'positions_and_heads': values} for row, values in hits.items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--identity', type=Path, required=True, help='Metadata only; redacted public identity works')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--tokens', type=Path, help='JSON array of token IDs')
    source.add_argument('--batch', type=Path, help='Saved native generation batch')
    parser.add_argument('--job', help='Exact job ID within --batch')
    parser.add_argument('--include-decoded', action='store_true', help='Also trace generated tokens actually fed back, excluding the final sample')
    parser.add_argument('--row', type=int, action='append', help='Report occurrences of this global row; repeat for multiple rows')
    args = parser.parse_args()
    identity_bytes = args.identity.read_bytes()
    metadata = json.loads(identity_bytes)['architecture_metadata']
    addressing = Addressing.from_metadata(metadata)
    if args.tokens:
        if args.job or args.include_decoded:
            parser.error('--job and --include-decoded require --batch')
        tokens = json.loads(args.tokens.read_text())
        if not isinstance(tokens, list):
            raise ValueError('Tokens file must contain a JSON array')
    else:
        if not args.job:
            parser.error('--batch requires --job')
        batch = json.loads(args.batch.read_text())
        matches = [j for j in batch['jobs'] if j['id'] == args.job]
        if len(matches) != 1:
            raise ValueError('Expected exactly one matching job')
        response = matches[0]['response']
        tokens = list(response['prompt_tokens'])
        if args.include_decoded:
            if response.get('schema') != 'ngramma.greedy-generation/v1':
                raise ValueError('Decoded-history convention requires the native greedy-generation schema')
            tokens += response['generated_tokens'][:-1]
    result = {'schema': 'ngramma.address-trace/v1', 'positions': len(tokens),
              'metadata_file_sha256': hashlib.sha256(identity_bytes).hexdigest(),
              'model_files_read': False, 'identity_authenticated': False,
              'scope': 'One contiguous text-token sequence; row arithmetic, not model inference.'}
    if args.row:
        result['occurrences'] = addressing.occurrences(tokens, args.row)
    else:
        result['trace'] = [{'position': i, 'token': t, 'rows': rows}
                           for i, (t, rows) in enumerate(zip(tokens, addressing.addresses(tokens)))]
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
