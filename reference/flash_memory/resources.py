"""Conservative allocation checks for the experimental CPU worker."""
from pathlib import Path
import resource

def check_budget(max_rss_gib=64,minimum_available_gib=24):
    info={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split())==3}
    status={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/self/status').read_text().splitlines() if line.startswith(('VmRSS:','VmHWM:'))}
    if status['VmRSS']>max_rss_gib*(1<<30):raise MemoryError('Worker resident-memory budget exceeded')
    if info['MemAvailable']<minimum_available_gib*(1<<30):raise MemoryError('Host available RAM below worker reserve')
    return {'rss_bytes':status['VmRSS'],'peak_rss_bytes':status.get('VmHWM',0),'available_bytes':info['MemAvailable']}
