"""Authenticate the recorded configuration and content of a local CPU fixture.

This binds artifacts to their recorded model label. Model shard authentication
is a separate prerequisite, not implied by this small-file validation.
"""
import json
from pathlib import Path
from .artifacts import file_hash


def validate_capture(directory, identity_sha256):
    directory = Path(directory)
    record = json.loads((directory/'capture-summary.json').read_text())
    required = {'identity_sha256': identity_sha256, 'device': 'CPU',
                'cache_type': 'f32', 'repack': True, 'flash_attention': False,
                'rope_overrides': False, 'context': 128, 'batch': 32, 'microbatch': 32}
    for key, value in required.items():
        if type(record.get(key)) is not type(value) or record[key] != value:
            raise ValueError(f'Capture configuration does not match the audited fixture: {key}')
    for filename, key in (('tensors.json', 'metadata_sha256'), ('logits.f32', 'logits_sha256')):
        if file_hash(directory/filename) != record[key]:
            raise ValueError(f'Capture content hash mismatch: {filename}')
    tokens = json.loads((directory/'tokens.json').read_text())
    if not isinstance(tokens, list) or any(type(token) is not int or not 0 <= token < 2**31 for token in tokens):
        raise ValueError('Require integer nonnegative int32 token IDs')
    if type(record.get('chunk_size')) is not int or record['tokens'] != tokens or not 0 < len(tokens) <= 32 or not len(tokens) <= record['chunk_size'] <= 32:
        raise ValueError('Require matching tokens in one complete prefill of at most 32 tokens')
    for key in ('lens_sha256', 'capture_script_sha256'):
        digest = record.get(key)
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in '0123456789abcdef' for char in digest):
            raise ValueError(f'Missing capture source identity: {key}')
    return record
