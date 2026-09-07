#!/usr/bin/env python3
"""Maintainer utility: hash reviewed publication files; never run automatically in CI."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
skip_parts = {'.git', '__pycache__', '.venv', '.pytest_cache'}
target = root/'data/SHA256SUMS.json'
files = {}
for path in sorted(root.rglob('*')):
    if path == target or not path.is_file() or any(p in skip_parts for p in path.relative_to(root).parts):
        continue
    if path.is_symlink():
        raise SystemExit(f'Refuse symlink: {path}')
    files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
target.write_text(json.dumps({'algorithm':'sha256','note':'All reviewed release files except this manifest; historical source hashes are separate.','files':files},indent=2)+'\n')
print(f'Wrote checksums for {len(files)} files.')
