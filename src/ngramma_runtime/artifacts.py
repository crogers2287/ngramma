from pathlib import Path
import hashlib
import json
import os
import tempfile

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()

def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def file_hash(path, offset=0, length=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(offset)
        left = length
        while left is None or left > 0:
            chunk = f.read(8 << 20 if left is None else min(8 << 20, left))
            if not chunk:
                if left:
                    raise ValueError(f"Truncated artifact: {path}")
                break
            h.update(chunk)
            if left is not None:
                left -= len(chunk)
    return h.hexdigest()

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(json.dumps(value, indent=2, ensure_ascii=False).encode() + b"\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

