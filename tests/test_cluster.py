"""
Processing on several nodes (cluster.py, --cluster_nodes): a coordinator and real mg_cluster_worker processes on
this machine, connected over TCP (localhost). The results must equal the single-process run with the same seed
(every neutron is seeded by its global index), also when a worker fails during the run.
"""
import os
import subprocess
import sys
import time

import numpy as np
import pytest

from test_run import COMMON_ARGV, _prepare_run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def localhost(monkeypatch):
    monkeypatch.setenv('MCSTAS_GISANS_CLUSTER_HOST', '127.0.0.1')


def _start_worker(address_file, processes, code=None):
    """mg_cluster_worker as a subprocess; code: Python run before the worker (e.g. a patch)."""
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, 'src') + os.pathsep + os.environ.get('PYTHONPATH', ''))
    prelude = (code or '') + "\nfrom mcstas_gisans.cluster import run_worker\n"
    script = prelude + f"run_worker({address_file!r}, {processes}, 60)\n"
    return subprocess.Popen([sys.executable, '-c', script], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, start_new_session=True)


def _finish(worker, timeout=60):
    """The worker's output and exit code; its process group (incl. orphaned pool processes) is killed afterwards."""
    import signal
    try:
        worker.wait(timeout=timeout)
    finally:
        try:
            os.killpg(worker.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    out = worker.stdout.read() if worker.stdout else ''
    return out, worker.returncode


def _particles_and_params(n=301):
    particles, params = _prepare_run(COMMON_ARGV + ["-i", "d22", "--wavelength_selected", "6.0"])
    return particles[:n], params


def test_two_nodes_reproduce_the_single_process_run(tmp_path, localhost):
    from mcstas_gisans import cluster
    from mcstas_gisans.run import process_particles, process_particles_parallelly
    particles, params = _particles_and_params()
    sequential = process_particles(particles, params)
    address = str(tmp_path / 'address.json')
    worker = _start_worker(address, 2)
    try:
        coordinator = cluster.start_coordinator(2, address, local_processes=2, timeout=60)
        assert len(coordinator.nodes) == 2 and coordinator.total_processes == 4
        distributed = process_particles_parallelly(particles, params, process_number=2)
        assert all(n.done > 0 for n in coordinator.nodes), "both nodes must have processed chunks"
        # a second simulation with the same connection
        again = process_particles_parallelly(particles, params, process_number=2)
    finally:
        cluster.stop_coordinator()
        out, code = _finish(worker)
    assert sequential['pixelHist'].sum() > 0
    for result in (distributed, again):
        np.testing.assert_allclose(result['pixelHist'], sequential['pixelHist'], rtol=1e-12, atol=0)
        np.testing.assert_allclose(result['pixelHistWeightsSquared'], sequential['pixelHistWeightsSquared'], rtol=1e-12, atol=0)
        assert result['directions'] == sequential['directions']
    assert code == 0, out
    assert not os.path.exists(address), "the address file (with the key) is removed at the end"


def test_a_failing_worker_node_is_replaced_by_the_others(tmp_path, localhost):
    """The worker process dies (os._exit) at its third chunk: its unfinished chunks are processed by the coordinator."""
    from mcstas_gisans import cluster
    from mcstas_gisans.run import process_particles, process_particles_parallelly
    particles, params = _particles_and_params()
    sequential = process_particles(particles, params)
    address = str(tmp_path / 'address.json')
    patch = ("import os, mcstas_gisans.cluster as c\n"
             "_orig = c.ProcessPoolExecutor.submit\n"
             "_n = [0]\n"
             "def submit(self, fn, *a, **k):\n"
             "    _n[0] += 1\n"
             "    if _n[0] == 3:\n"
             "        os._exit(3)\n"
             "    return _orig(self, fn, *a, **k)\n"
             "c.ProcessPoolExecutor.submit = submit\n")
    worker = _start_worker(address, 2, patch)
    try:
        cluster.start_coordinator(2, address, local_processes=2, timeout=60)
        distributed = process_particles_parallelly(particles, params, process_number=2)
    finally:
        cluster.stop_coordinator()
        _finish(worker)
    np.testing.assert_allclose(distributed['pixelHist'], sequential['pixelHist'], rtol=1e-12, atol=0)


def test_a_chunk_error_on_a_worker_is_raised(tmp_path, localhost):
    from mcstas_gisans import cluster
    from mcstas_gisans.run import process_particles_parallelly
    particles, params = _particles_and_params(60)
    address = str(tmp_path / 'address.json')
    patch = ("import multiprocessing; multiprocessing.set_start_method('fork', force=True)\n"  # the patch reaches the pool
             "import mcstas_gisans.run as r\n"
             "def process_particles(*a, **k):\n"
             "    raise ValueError('broken chunk')\n"
             "r.process_particles = process_particles\n")
    worker = _start_worker(address, 1, patch)
    try:
        cluster.start_coordinator(2, address, local_processes=0, timeout=60)  # all chunks on the worker
        with pytest.raises(RuntimeError, match='broken chunk'):
            process_particles_parallelly(particles, params, process_number=1)
    finally:
        cluster.stop_coordinator()
        _finish(worker)


def test_missing_workers_time_out(tmp_path, localhost):
    from mcstas_gisans import cluster
    t0 = time.time()
    with pytest.raises(RuntimeError, match='0 of 1 worker'):
        cluster.start_coordinator(2, str(tmp_path / 'address.json'), local_processes=1, timeout=2)
    assert time.time() - t0 < 30
    cluster.stop_coordinator()


def test_cluster_options():
    from mcstas_gisans.run_cli import create_argparser, parse_args
    parser = create_argparser()
    argv = sys.argv
    try:
        sys.argv = ["run", "x.mcpl", "-i", "d22", "--wavelength_selected", "6", "--cluster_nodes", "3",
                    "--cluster_address_file", "/tmp/a.json"]
        args = parse_args(parser)
        assert args.cluster_nodes == 3 and args.cluster_address_file == "/tmp/a.json"
        sys.argv = ["run", "x.mcpl", "-i", "d22", "--wavelength_selected", "6", "--cluster_nodes", "2", "--no_parallel"]
        with pytest.raises(SystemExit):
            parse_args(parser)
    finally:
        sys.argv = argv


def test_cluster_options_do_not_change_the_fit_settings():
    from mcstas_gisans.fit_restart import NON_RESULT_OPTIONS
    assert {'cluster_nodes', 'cluster_address_file', 'cluster_timeout'} <= NON_RESULT_OPTIONS
