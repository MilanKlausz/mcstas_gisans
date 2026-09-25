import os
import mcstas_gisans.hardware as hardware
from mcstas_gisans.hardware import get_available_cores


def test_slurm_allocation_is_respected(monkeypatch):
  monkeypatch.setenv("SLURM_CPUS_PER_TASK", "7")
  assert get_available_cores() == 7


def test_at_least_one_core_is_reported(monkeypatch):
  monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
  assert get_available_cores() >= 1
  # detection failing everywhere: fall back to half of the logical cores, at least one
  monkeypatch.setattr(hardware.subprocess, "check_output", lambda *a, **k: (_ for _ in ()).throw(OSError()))
  monkeypatch.setattr(hardware.sys, "platform", "unknown")
  monkeypatch.setattr(hardware.multiprocessing, "cpu_count", lambda: 1)
  if hasattr(os, "sched_getaffinity"):
    monkeypatch.setattr(hardware.os, "sched_getaffinity", lambda pid: {0})
  assert get_available_cores() == 1

