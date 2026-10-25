"""
Parallel processing over several cluster nodes (mg_run / mg_fit --cluster_nodes).

The incident neutrons of one simulation are split into chunks (process_particles_parallelly), and the chunks are
processed by the CPU processes of all nodes of the job: this process (the coordinator) runs a local process pool, and
every other node runs one worker (mg_cluster_worker) with its own process pool, connected to the coordinator over
TCP (multiprocessing.connection, authenticated with a random key). The nodes pull chunks as their processes become
free, so faster nodes and cheaper chunks balance automatically; the partial detector histograms are summed in chunk
order. Every neutron is seeded by its global index (process_particles), so the result does not depend on the
number of nodes or on the chunking: a run on several nodes reproduces the single-node run with the same --seed
(up to the rounding of the summation order).

Start-up: the coordinator listens on a free port and writes the address and key to --cluster_address_file (on a
file system shared by the nodes, readable only by the user); the workers wait for that file, connect, and process
chunks until the coordinator closes the connection. In a Slurm job with N nodes::

    srun --relative=1 --nodes=$((SLURM_NNODES-1)) --ntasks=$((SLURM_NNODES-1)) --ntasks-per-node=1 \\
        mg_cluster_worker --address_file ADDR --processes 112 &
    mg_fit ... --parallel_processes 112 --cluster_nodes $SLURM_NNODES --cluster_address_file ADDR

If a worker node fails, its unfinished chunks are given to the other nodes (the run goes on with fewer nodes).
Not supported: time-of-flight event files (TOF instruments without --wavelength_slice), whose temporary files would
stay on the worker nodes.
"""
import json
import os
import pickle
import queue
import secrets
import socket
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from multiprocessing.connection import Client, Listener
from typing import Any, Callable, Dict, List, Optional, Sequence

_COORDINATOR = None


def get_coordinator():
    """The running coordinator of this process (None: single-node processing)."""
    return _COORDINATOR


def _write_address_file(path: str, info: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as f:
        json.dump(info, f)
    os.replace(tmp, path)


def _function_name(func: Callable) -> str:
    """'module:name' of a module-level function; a function of the script run with python -m gets its module name."""
    module = func.__module__
    if module == '__main__':
        spec = getattr(sys.modules['__main__'], '__spec__', None)
        module = spec.name if spec is not None else module
    return f"{module}:{func.__name__}"


def _pack(value: Any) -> Any:
    """Results sent over the network: float arrays with few non-zero values (a chunk of neutrons hits only part of the
    detector) as (shape, dtype, indices, values) - exact, and much smaller than the dense arrays."""
    import numpy as np
    if isinstance(value, dict):
        return {k: _pack(v) for k, v in value.items()}
    if isinstance(value, np.ndarray) and value.dtype.kind == 'f' and value.size >= 1024:
        nonzero = np.flatnonzero(value)
        if nonzero.size < value.size // 3:
            return ('__sparse__', value.shape, value.dtype.str, nonzero.astype(np.int32), value.ravel()[nonzero])
    return value


def _unpack(value: Any) -> Any:
    import numpy as np
    if isinstance(value, dict):
        return {k: _unpack(v) for k, v in value.items()}
    if isinstance(value, tuple) and len(value) == 5 and value[0] == '__sparse__':
        _, shape, dtype, nonzero, values = value
        dense = np.zeros(int(np.prod(shape)), dtype=np.dtype(dtype))
        dense[nonzero] = values
        return dense.reshape(shape)
    return value


_SHARED_CACHE: Dict[str, Any] = {}


def _call_with_shared(func_name: str, shared_path: str, shared_index: int, args: tuple) -> Any:
    """Run func(*args) with the shared argument (the same for all chunks of a call, e.g. the simulation parameters)
    inserted at shared_index; the shared argument is loaded once per process and call from the node-local file."""
    if shared_path not in _SHARED_CACHE:
        _SHARED_CACHE.clear()
        with open(shared_path, 'rb') as f:
            _SHARED_CACHE[shared_path] = pickle.load(f)
    full = list(args)
    full.insert(shared_index, _SHARED_CACHE[shared_path])
    return _resolve(func_name)(*full)


def _write_shared(payload: bytes, tag: str) -> str:
    fd, path = tempfile.mkstemp(prefix=f'mcstas_gisans_shared_{tag}_', suffix='.pkl')
    with os.fdopen(fd, 'wb') as f:
        f.write(payload)
    return path


class _Node:
    """One node of the run as seen by the coordinator: a remote worker (conn) or the local pool (conn None)."""

    def __init__(self, name: str, processes: int, conn=None):
        self.name, self.processes, self.conn = name, processes, conn
        self.done = 0
        self.alive = True


class Coordinator:
    """
    Accepts the workers and distributes chunk tasks over them and a local process pool (see the module docstring).

    Parameters
    ----------
    n_nodes : int
        Number of nodes including this one; n_nodes - 1 workers must connect.
    address_file : str
        File for the address and key (shared file system).
    local_processes : int
        Size of the local process pool (0: this node only coordinates).
    timeout : float
        Seconds to wait for the workers to connect.
    """

    def __init__(self, n_nodes: int, address_file: str, local_processes: int, timeout: float = 600.0):
        self.address_file = address_file
        self.authkey = secrets.token_bytes(32)
        self.listener = Listener(('0.0.0.0', 0), authkey=self.authkey)
        # the address the workers connect to (MCSTAS_GISANS_CLUSTER_HOST overrides, e.g. 127.0.0.1 for tests)
        host = os.environ.get('MCSTAS_GISANS_CLUSTER_HOST') or socket.gethostname()
        _write_address_file(address_file, {'host': host, 'port': self.listener.address[1], 'authkey': self.authkey.hex(),
                                           'pid': os.getpid(), 'created': time.time()})
        print(f"Cluster: coordinator on {host}:{self.listener.address[1]}, waiting for {n_nodes - 1} worker node(s) "
              f"(address file {address_file})", flush=True)
        self.nodes: List[_Node] = []
        if local_processes > 0:
            self.nodes.append(_Node(f"{host} (coordinator)", local_processes))
        self._local_pool = ProcessPoolExecutor(max_workers=local_processes) if local_processes > 0 else None
        self._lock = threading.Lock()
        self.chunks_per_process = 4
        self._accept(n_nodes - 1, timeout)
        total = sum(n.processes for n in self.nodes)
        print(f"Cluster: {len(self.nodes)} node(s) with {total} processes: " +
              ", ".join(f"{n.name} {n.processes}" for n in self.nodes), flush=True)
    def _accept(self, n_workers: int, timeout: float) -> None:
        accepted: List[Any] = []
        error: List[BaseException] = []

        def accept_all():
            try:
                while len(accepted) < n_workers:
                    conn = self.listener.accept()
                    hello = conn.recv()
                    accepted.append((conn, hello))
                    print(f"Cluster: worker {hello['host']} connected ({hello['processes']} processes)", flush=True)
            except BaseException as err:  # listener closed after the timeout
                error.append(err)
        thread = threading.Thread(target=accept_all, daemon=True)
        thread.start()
        thread.join(timeout)
        if len(accepted) < n_workers:
            self.listener.close()
            raise RuntimeError(f"Cluster: only {len(accepted)} of {n_workers} worker node(s) connected within {timeout:g} s "
                               f"(are the mg_cluster_worker processes started with --address_file {self.address_file}?)")
        for conn, hello in accepted:
            self.nodes.append(_Node(hello['host'], int(hello['processes']), conn))

    @property
    def total_processes(self) -> int:
        return sum(n.processes for n in self.nodes if n.alive)

    def map(self, func: Callable, tasks: Sequence[tuple], shared: Any = None, shared_index: int = 0) -> List[Any]:
        """``func(*task)`` for every task on the nodes; results in task order. func must be importable by the workers
        (a module-level function of mcstas_gisans). The first task error is raised here. With `shared`, every task
        lacks one argument, `shared`, inserted at shared_index: it is pickled once and sent once to every node
        (instead of with every chunk), and each process loads it once."""
        self._call = getattr(self, '_call', 0) + 1
        call = self._call
        func_name = _function_name(func)
        payload = pickle.dumps(shared, protocol=pickle.HIGHEST_PROTOCOL) if shared is not None else None
        local_shared = _write_shared(payload, f'c{os.getpid()}_{call}') if payload is not None and self._local_pool else None
        pending: "queue.Queue[int]" = queue.Queue()
        for i in range(len(tasks)):
            pending.put(i)
        results: Dict[int, Any] = {}
        errors: List[str] = []
        t0 = time.time()

        def finish(i, value):
            with self._lock:
                results.setdefault(i, value)

        def fail(message):
            with self._lock:
                errors.append(message)

        def serve_local(node: _Node):
            in_flight = {}
            while True:
                while len(in_flight) < 2 * node.processes and not errors:  # prefetch: the pool queue keeps every process busy
                    try:
                        i = pending.get_nowait()
                    except queue.Empty:
                        break
                    fut = (self._local_pool.submit(_call_with_shared, func_name, local_shared, shared_index, tasks[i])
                           if local_shared else self._local_pool.submit(func, *tasks[i]))
                    in_flight[fut] = i
                if not in_flight:
                    return
                done, _ = wait(list(in_flight), timeout=1.0, return_when=FIRST_COMPLETED)
                for fut in done:
                    i = in_flight.pop(fut)
                    try:
                        finish(i, fut.result())
                        node.done += 1
                    except BaseException:
                        fail(f"task {i} on {node.name}:\n{traceback.format_exc()}")

        def serve_remote(node: _Node):
            in_flight = set()
            try:
                if payload is not None:
                    node.conn.send(('shared', (call, -1), shared_index, payload))
                while True:
                    while len(in_flight) < 2 * node.processes and not errors:  # a little prefetch keeps the pool busy
                        try:
                            i = pending.get_nowait()
                        except queue.Empty:
                            break
                        node.conn.send(('task', (call, i), func_name, tasks[i]))
                        in_flight.add(i)
                    if not in_flight:
                        return
                    if not node.conn.poll(1.0):
                        continue
                    kind, (c, i), value = node.conn.recv()
                    if c != call:  # a late result of an earlier, failed call
                        continue
                    in_flight.discard(i)
                    if kind == 'result':
                        finish(i, _unpack(value))
                        node.done += 1
                    else:
                        fail(f"task {i} on {node.name}:\n{value}")
            except (EOFError, OSError, BrokenPipeError) as err:
                node.alive = False
                print(f"Cluster: lost worker {node.name} ({err!r}); its {len(in_flight)} unfinished chunk(s) go to the "
                      f"other nodes", flush=True)
                for i in in_flight:
                    pending.put(i)
            except BaseException:
                fail(f"coordinator thread of {node.name}:\n{traceback.format_exc()}")

        def guarded(serve, node):
            try:
                serve(node)
            except BaseException:
                fail(f"coordinator thread of {node.name}:\n{traceback.format_exc()}")

        for n in self.nodes:
            n.done = 0
        threads = [threading.Thread(target=guarded, args=(serve_local if n.conn is None else serve_remote, n), daemon=True)
                   for n in self.nodes if n.alive]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if local_shared:
            os.remove(local_shared)
        if errors:
            raise RuntimeError("Cluster: a chunk failed: " + errors[0])
        # a node that failed after the others had drained the queue leaves chunks: run them on the remaining nodes
        missing = [i for i in range(len(tasks)) if i not in results]
        if missing:
            if not any(n.alive for n in self.nodes) or len(missing) == len(tasks):
                raise RuntimeError(f"Cluster: {len(missing)} chunk(s) could not be processed (no working node left)")
            print(f"Cluster: {len(missing)} chunk(s) left after a node failure, processed again", flush=True)
            for i, value in zip(missing, self.map(func, [tasks[i] for i in missing], shared, shared_index)):
                results[i] = value
        print(f"Cluster: {len(tasks)} chunks in {time.time() - t0:.1f} s" +
              (f", shared {len(payload) / 1e6:.2f} MB" if payload is not None else "") + " (" +
              ", ".join(f"{n.name.split('.')[0]} {n.done}" for n in self.nodes if n.alive) + ")", flush=True)
        return [results[i] for i in range(len(tasks))]

    def close(self) -> None:
        for n in self.nodes:
            if n.conn is not None and n.alive:
                try:
                    n.conn.send(('stop', None, None, None))
                    n.conn.close()
                except (OSError, BrokenPipeError):
                    pass
        if self._local_pool is not None:
            self._local_pool.shutdown(wait=True)
        self.listener.close()
        try:
            os.remove(self.address_file)
        except OSError:
            pass


def start_coordinator(n_nodes: int, address_file: Optional[str], local_processes: int, timeout: float = 600.0,
                      chunks_per_process: int = 4) -> Coordinator:
    """Start the coordinator of this process (see Coordinator), closed at exit."""
    global _COORDINATOR
    import atexit
    if not address_file:
        job = os.environ.get('SLURM_JOB_ID', str(os.getpid()))
        address_file = os.path.join(os.path.expanduser('~'), '.mcstas_gisans_cluster', f'address_{job}.json')
    _COORDINATOR = Coordinator(n_nodes, address_file, local_processes, timeout)
    _COORDINATOR.chunks_per_process = max(1, int(chunks_per_process))
    atexit.register(stop_coordinator)
    return _COORDINATOR


def stop_coordinator() -> None:
    global _COORDINATOR
    if _COORDINATOR is not None:
        _COORDINATOR.close()
        _COORDINATOR = None


def setup_from_args(args: Any) -> Optional[Coordinator]:
    """Start the coordinator if args.cluster_nodes > 1 (mg_run / mg_fit)."""
    n_nodes = getattr(args, 'cluster_nodes', 1) or 1
    if n_nodes <= 1:
        return None
    from .hardware import get_available_cores
    local = args.parallel_processes if args.parallel_processes else max(1, get_available_cores() - 1)
    return start_coordinator(n_nodes, getattr(args, 'cluster_address_file', None), local,
                             getattr(args, 'cluster_timeout', 600.0), getattr(args, 'cluster_chunks_per_process', 4))


# ---------------------------------------------------------------------------------------------------------------- worker

def _resolve(func_name: str) -> Callable:
    import importlib
    module, name = func_name.split(':')
    if not module.startswith('mcstas_gisans'):
        raise ValueError(f"refusing to run {func_name}: only mcstas_gisans functions")
    return getattr(importlib.import_module(module), name)


def run_worker(address_file: str, processes: int, wait_timeout: float = 600.0) -> None:
    """Connect to the coordinator of --address_file and process its chunk tasks with a pool of `processes` until it
    closes the connection."""
    t0 = time.time()
    while True:
        try:
            with open(address_file) as f:
                info = json.load(f)
            break
        except (OSError, ValueError):
            if time.time() - t0 > wait_timeout:
                raise RuntimeError(f"mg_cluster_worker: no coordinator address in {address_file} after {wait_timeout:g} s")
            time.sleep(0.5)
    conn = Client((info['host'], int(info['port'])), authkey=bytes.fromhex(info['authkey']))
    conn.send({'host': socket.gethostname(), 'processes': processes})
    print(f"mg_cluster_worker on {socket.gethostname()}: connected to {info['host']}:{info['port']} with {processes} "
          f"processes", flush=True)
    pool = ProcessPoolExecutor(max_workers=processes)
    send_lock = threading.Lock()
    funcs: Dict[str, Callable] = {}
    n_done = 0

    def send(message):
        with send_lock:
            conn.send(message)

    def done_callback(key):
        def callback(fut):
            nonlocal n_done
            try:
                send(('result', key, _pack(fut.result())))
                n_done += 1
            except BaseException:
                try:
                    send(('error', key, traceback.format_exc()))
                except (OSError, BrokenPipeError):
                    pass
        return callback

    shared: Dict[int, tuple] = {}  # call -> (shared_index, node-local file of the shared argument)

    def drop_shared(keep=None):
        for c in [c for c in shared if c != keep]:
            try:
                os.remove(shared.pop(c)[1])
            except OSError:
                pass

    try:
        while True:
            try:
                kind, key, func_name, task = conn.recv()
            except EOFError:
                break
            if kind == 'stop':
                break
            call = key[0]
            if kind == 'shared':  # ('shared', (call, -1), shared_index, pickled shared argument)
                drop_shared()
                shared[call] = (func_name, _write_shared(task, f'w{os.getpid()}_{call}'))
                continue
            if call in shared:
                index, path = shared[call]
                fut = pool.submit(_call_with_shared, func_name, path, index, task)
            else:
                if func_name not in funcs:
                    funcs[func_name] = _resolve(func_name)
                fut = pool.submit(funcs[func_name], *task)
            fut.add_done_callback(done_callback(key))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        drop_shared()
        try:
            conn.close()
        except OSError:
            pass
    print(f"mg_cluster_worker on {socket.gethostname()}: finished ({n_done} chunks in {time.time() - t0:.0f} s)", flush=True)


def worker_main() -> None:
    """mg_cluster_worker: the worker of one node of a multi-node mg_run / mg_fit (see the module docstring)."""
    import argparse
    from .hardware import get_available_cores
    parser = argparse.ArgumentParser(description="Worker of one node of a multi-node mg_run / mg_fit (--cluster_nodes): "
                                                 "connects to the coordinator named in --address_file and processes its "
                                                 "neutron chunks with a pool of processes.")
    parser.add_argument('--address_file', required=True, help='The --cluster_address_file of the mg_run / mg_fit coordinator.')
    parser.add_argument('-p', '--processes', type=int, default=None,
                        help='Number of processes on this node (default: number of CPU cores - 1).')
    parser.add_argument('--wait_timeout', type=float, default=600.0, help='Seconds to wait for the address file.')
    args = parser.parse_args()
    run_worker(args.address_file, args.processes or max(1, get_available_cores() - 1), args.wait_timeout)


if __name__ == '__main__':
    worker_main()
