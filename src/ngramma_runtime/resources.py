"""Linux resident-memory guard for local CPU diagnostics (no allocation)."""
from pathlib import Path
import math
import os


def check_budget(max_rss_gib=None, minimum_available_gib=None):
    maximum = float(os.environ.get("NGRAMMA_MAX_RSS_GIB", "64")) if max_rss_gib is None else float(max_rss_gib)
    reserve = float(os.environ.get("NGRAMMA_MIN_AVAILABLE_GIB", "24")) if minimum_available_gib is None else float(minimum_available_gib)
    if not math.isfinite(maximum) or maximum <= 0 or not math.isfinite(reserve) or reserve < 0:
        raise ValueError("Memory limits require finite max_rss_gib > 0 and minimum_available_gib >= 0")
    try:
        info = {line.split(':')[0]: int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split()) == 3}
        status = {line.split(':')[0]: int(line.split()[1])*1024 for line in Path('/proc/self/status').read_text().splitlines() if line.startswith(('VmRSS:', 'VmHWM:'))}
        measured = {'rss_bytes': status['VmRSS'], 'peak_rss_bytes': status.get('VmHWM', 0), 'available_bytes': info['MemAvailable']}
    except (OSError, KeyError, ValueError) as exc:
        raise RuntimeError("Memory guard requires Linux /proc/meminfo and /proc/self/status; cannot safely admit a model forward on this host") from exc
    if measured['rss_bytes'] > maximum*(1 << 30):
        raise MemoryError('Worker resident-memory budget exceeded')
    if measured['available_bytes'] < reserve*(1 << 30):
        raise MemoryError('Host available RAM below worker reserve')
    return measured
