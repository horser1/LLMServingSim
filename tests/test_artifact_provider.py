import multiprocessing
import os
import tempfile
import unittest
from pathlib import Path

from serving.core.artifact_provider import GraphArtifactProvider


class FakeGraphArtifactProvider(GraphArtifactProvider):
    def __init__(self, *args, convert_started=None, release_convert=None,
                 convert_count=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.convert_started = convert_started
        self.release_convert = release_convert
        self.convert_count = convert_count

    def _convert_inprocess(
            self, trace_path, output_prefix, num_npus, npu_offset,
            local_offloading):
        execution_type = Path(trace_path).read_bytes().split(None, 1)[0]
        count = int(num_npus) * (2 if execution_type == b"PREFILL" else 1)
        if self.convert_count is not None:
            with self.convert_count.get_lock():
                self.convert_count.value += 1
        if self.convert_started is not None:
            self.convert_started.set()
        if self.release_convert is not None:
            self.release_convert.wait(timeout=5)
        for index in range(count):
            rank = int(npu_offset) + index
            Path(f"{output_prefix}.{rank}.et").write_bytes(
                f"graph-{rank}".encode("ascii"))


def _write_fake_converter(root):
    converter = root / "src" / "converter"
    converter.mkdir(parents=True, exist_ok=True)
    (converter / "llm_converter.py").write_text(
        "# fake converter identity\n", encoding="utf-8")


def _provide_worker(chakra_root, cache, output, trace, convert_started,
                    release_convert, convert_count, queue):
    try:
        provider = FakeGraphArtifactProvider(
            chakra_root, cache_dir=cache, convert_started=convert_started,
            release_convert=release_convert, convert_count=convert_count)
        queue.put(("ok", provider.provide(trace, output, num_npus=1)))
    except Exception as error:
        queue.put(("error", repr(error)))


def _hold_entry_lock_worker(chakra_root, cache, key, locked, release, queue):
    try:
        provider = GraphArtifactProvider(chakra_root, cache_dir=cache)
        with provider._entry_lock(key):
            locked.set()
            release.wait(timeout=5)
        queue.put(("ok", None))
    except Exception as error:
        queue.put(("error", repr(error)))


class GraphArtifactProviderTest(unittest.TestCase):
    def make_provider(self, root, cache):
        _write_fake_converter(root)
        return FakeGraphArtifactProvider(
            str(root), cache_dir=str(cache))

    def test_prefill_caches_and_materializes_paired_et_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider = self.make_provider(root / "chakra", root / "cache")
            output = root / "run" / "llm"

            hit = provider.provide(
                b"PREFILL\t\tmodel_parallel_NPU_group: 1\n0\nheader\n",
                str(output), num_npus=1)

            self.assertFalse(hit)
            self.assertEqual((root / "run" / "llm.0.et").read_bytes(), b"graph-0")
            self.assertEqual((root / "run" / "llm.1.et").read_bytes(), b"graph-1")

    def test_concurrent_processes_share_one_immutable_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chakra = root / "chakra"
            cache = root / "cache"
            _write_fake_converter(chakra)
            trace = b"COLOCATED\t\tmodel_parallel_NPU_group: 1\n0\nheader\n"
            ctx = multiprocessing.get_context("fork")
            convert_started = ctx.Event()
            release_convert = ctx.Event()
            convert_count = ctx.Value("i", 0)
            queue = ctx.Queue()

            first = ctx.Process(
                target=_provide_worker,
                args=(str(chakra), str(cache), str(root / "run-0" / "llm"),
                      trace, convert_started, release_convert, convert_count,
                      queue))
            first.start()
            self.assertTrue(convert_started.wait(timeout=5))

            second = ctx.Process(
                target=_provide_worker,
                args=(str(chakra), str(cache), str(root / "run-1" / "llm"),
                      trace, None, None, convert_count, queue))
            second.start()
            self.assertEqual(convert_count.value, 1)
            release_convert.set()
            first.join(timeout=5)
            second.join(timeout=5)
            self.assertFalse(first.is_alive())
            self.assertFalse(second.is_alive())

            results = [queue.get(timeout=1), queue.get(timeout=1)]
            self.assertEqual([kind for kind, _ in results], ["ok", "ok"])
            self.assertEqual(sorted(value for _, value in results), [False, True])
            self.assertEqual((root / "run-0" / "llm.0.et").read_bytes(), b"graph-0")
            self.assertEqual((root / "run-1" / "llm.0.et").read_bytes(), b"graph-0")
            entries = [
                path for path in cache.iterdir()
                if path.is_dir() and not path.name.startswith(".")
            ]
            self.assertEqual(len(entries), 1)
            self.assertEqual(
                [name for name in os.listdir(cache) if ".tmp." in name],
                [],
            )

    def test_prune_skips_entry_locked_by_another_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider = self.make_provider(root / "chakra", root / "cache")
            trace_a = b"COLOCATED\t\tmodel_parallel_NPU_group: 1\n0\nA\n"
            trace_b = b"COLOCATED\t\tmodel_parallel_NPU_group: 1\n0\nB\n"
            provider.provide(trace_a, str(root / "run-a" / "llm"), num_npus=1)
            provider.provide(trace_b, str(root / "run-b" / "llm"), num_npus=1)
            key_a, _ = provider._key(trace_a, 1, 0, False)
            key_b, _ = provider._key(trace_b, 1, 0, False)
            entry_a = root / "cache" / key_a
            entry_b = root / "cache" / key_b
            ctx = multiprocessing.get_context("fork")
            locked = ctx.Event()
            release = ctx.Event()
            queue = ctx.Queue()
            locker = ctx.Process(
                target=_hold_entry_lock_worker,
                args=(str(root / "chakra"), str(root / "cache"), key_a,
                      locked, release, queue))
            locker.start()
            self.assertTrue(locked.wait(timeout=5))

            provider.max_bytes = 1
            provider._prune(str(entry_b))
            self.assertTrue(entry_a.exists())

            release.set()
            locker.join(timeout=5)
            self.assertFalse(locker.is_alive())
            self.assertEqual(queue.get(timeout=1), ("ok", None))
            provider._prune(str(entry_b))
            self.assertFalse(entry_a.exists())
            self.assertTrue(entry_b.exists())


if __name__ == "__main__":
    unittest.main()
