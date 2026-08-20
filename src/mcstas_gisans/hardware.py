"""
Utility module for detecting hardware and system resources.
"""

import os
import sys
import subprocess
import multiprocessing

def get_available_cores():
    """
    Intelligently determines the number of available physical cores,
    respecting SLURM allocations and ignoring hyperthreads when possible.
    """
    # 1. Check SLURM allocation first
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    
    # 2. Check cgroups/OS affinity (returns logical cores allowed for this process)
    allowed_logical_cores = None
    if hasattr(os, 'sched_getaffinity'):
        allowed_logical_cores = len(os.sched_getaffinity(0))

    # 3. Try to determine physical cores
    physical_cores = None
    try:
        if sys.platform == 'darwin':
            physical_cores = int(subprocess.check_output(['sysctl', '-n', 'hw.physicalcpu']).strip())
        elif sys.platform == 'linux':
            core_info = set()
            with open('/proc/cpuinfo', 'r') as f:
                phys_id = None
                for line in f:
                    if line.startswith('physical id'):
                        phys_id = line.split(':')[1].strip()
                    elif line.startswith('core id'):
                        core_id = line.split(':')[1].strip()
                        if phys_id is not None:
                            core_info.add((phys_id, core_id))
            if core_info:
                physical_cores = len(core_info)
    except Exception:
        pass

    # Reconcile affinity and physical cores
    if physical_cores is not None:
        total_logical = multiprocessing.cpu_count()
        if allowed_logical_cores is not None and allowed_logical_cores < total_logical:
            # We are restricted by cgroups. If the system has SMT (logical > physical), assume affinity is also logical.
            if total_logical > physical_cores:
                return max(1, allowed_logical_cores // 2)
            return allowed_logical_cores
        return physical_cores
    
    # Fallback if detection fails
    if allowed_logical_cores:
        return max(1, allowed_logical_cores // 2) # Assume hyperthreading is active on HPC
    return max(1, multiprocessing.cpu_count() // 2)
