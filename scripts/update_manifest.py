#!/usr/bin/env python3
"""Maintainer utility: hash reviewed publication files; never run automatically in CI."""
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
target = root/'data/SHA256SUMS.json'
files = {}
# Hash the reviewed Git index's file set, not concurrent untracked experiments
# or ignored raw captures. Stage intended new files before refreshing this file.
tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
for name in sorted(n for n in tracked if n):
    path = root / name
    if path == target:
        continue
    if not path.is_file():
        raise SystemExit(f'Tracked file missing: {name}; stage removals first.')
    if path.is_symlink():
        raise SystemExit(f'Refuse symlink: {path}')
    files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
target.write_text(json.dumps({'algorithm':'sha256','note':'All reviewed release files except this manifest; historical source hashes are separate.','files':files},indent=2)+'\n')
print(f'Wrote checksums for {len(files)} files.')
