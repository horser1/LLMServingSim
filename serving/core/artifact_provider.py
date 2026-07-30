"""Exact Chakra graph artifacts: in-process conversion and content cache."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from time import time


_CACHE_SCHEMA = 1
_PRUNE_INTERVAL = 1024


class GraphArtifactProvider:
    """Materialize a workload prefix from canonical trace bytes.

    Cache identity covers every converter input which can affect graph bytes.
    A hit only avoids conversion; ASTRA-Sim still receives and executes the
    exact same per-token graph, so it is valid in the default fidelity mode.
    """

    def __init__(self, chakra_root, cache_dir=None, enable_cache=True,
                 converter_mode="inprocess", telemetry=None, max_bytes=None):
        self.chakra_root = os.path.abspath(chakra_root)
        self.cache_dir = os.path.abspath(cache_dir) if cache_dir else None
        self.enable_cache = bool(enable_cache and self.cache_dir)
        self.converter_mode = converter_mode
        self.telemetry = telemetry
        self.max_bytes = max_bytes
        self._entries_since_prune = 0
        if self.enable_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _converter_version(self):
        source = os.path.join(self.chakra_root, "src", "converter", "llm_converter.py")
        with open(source, "rb") as f:
            source_hash = hashlib.sha256(f.read()).hexdigest()
        return {
            "cache_schema": _CACHE_SCHEMA,
            "chakra_schema": "1.0.2-chakra.0.0.4",
            "llm_converter_sha256": source_hash,
        }

    def _key(self, trace_bytes, num_npus, npu_offset, local_offloading):
        identity = {
            "converter": self._converter_version(),
            "num_npus": int(num_npus),
            "npu_offset": int(npu_offset),
            "local_offloading": bool(local_offloading),
            "trace_sha256": hashlib.sha256(trace_bytes).hexdigest(),
        }
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(trace_bytes + b"\0" + canonical).hexdigest(), identity

    @staticmethod
    def _manifest_path(entry):
        return os.path.join(entry, "manifest.json")

    def _valid_entry(self, entry, key):
        try:
            with open(self._manifest_path(entry), "r", encoding="utf-8") as f:
                manifest = json.load(f)
            if manifest.get("key") != key or manifest.get("schema") != _CACHE_SCHEMA:
                return None
            for name, digest in manifest["files"].items():
                path = os.path.join(entry, name)
                with open(path, "rb") as f:
                    if hashlib.sha256(f.read()).hexdigest() != digest:
                        return None
            return manifest
        except (FileNotFoundError, KeyError, OSError, ValueError, json.JSONDecodeError):
            return None

    @staticmethod
    def _atomic_link_or_copy(source, target):
        temp_target = f"{target}.tmp.{os.getpid()}"
        try:
            try:
                os.link(source, temp_target)
            except OSError:
                shutil.copy2(source, temp_target)
            os.replace(temp_target, target)
        finally:
            try:
                os.unlink(temp_target)
            except FileNotFoundError:
                pass

    def _materialize(self, entry, output_prefix, manifest):
        os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
        for name in manifest["files"]:
            self._atomic_link_or_copy(os.path.join(entry, name), f"{output_prefix}{name[3:]}")

    def _convert_inprocess(self, trace_path, output_prefix, num_npus, npu_offset, local_offloading):
        if self.chakra_root not in sys.path:
            sys.path.insert(0, self.chakra_root)
        from chakra.src.converter.llm_converter import LLMConverter
        LLMConverter(trace_path, output_prefix, num_npus, npu_offset, local_offloading).convert()

    def _convert_subprocess(self, trace_path, output_prefix, num_npus, npu_offset, local_offloading):
        cmd = [
            sys.executable, "-m", "chakra.src.converter.converter", "LLM",
            "--input", trace_path, "--output", output_prefix,
            "--num-npus", str(num_npus), "--npu-offset", str(npu_offset),
        ]
        if local_offloading:
            cmd.append("--local-offloading")
        subprocess.run(cmd, cwd=self.chakra_root, text=True, check=True)

    def _build_entry(self, entry, trace_bytes, num_npus, npu_offset, local_offloading, key, identity):
        parent = os.path.dirname(entry)
        tmp = tempfile.mkdtemp(prefix=f".{key}.tmp.", dir=parent)
        try:
            trace_path = os.path.join(tmp, "trace.txt")
            output_prefix = os.path.join(tmp, "llm")
            with open(trace_path, "wb") as f:
                f.write(trace_bytes)
            if self.converter_mode == "subprocess":
                self._convert_subprocess(trace_path, output_prefix, num_npus, npu_offset, local_offloading)
            else:
                self._convert_inprocess(trace_path, output_prefix, num_npus, npu_offset, local_offloading)
            files = {}
            for name in sorted(os.listdir(tmp)):
                if name.startswith("llm.") and name.endswith(".et"):
                    with open(os.path.join(tmp, name), "rb") as f:
                        files[name] = hashlib.sha256(f.read()).hexdigest()
            if len(files) != int(num_npus):
                raise RuntimeError(
                    f"Chakra conversion produced {len(files)} ET files, expected {num_npus}.")
            manifest = {
                "schema": _CACHE_SCHEMA, "key": key, "identity": identity,
                "created_at": time(), "files": files,
            }
            with open(self._manifest_path(tmp), "w", encoding="utf-8") as f:
                json.dump(manifest, f, sort_keys=True)
            try:
                os.replace(tmp, entry)
                tmp = None
            except FileExistsError:
                # Another process populated the same immutable content key.
                pass
        finally:
            if tmp is not None:
                shutil.rmtree(tmp, ignore_errors=True)

    def _prune(self, protected_entry):
        """Bound cache growth with an LRU-by-directory-mtime policy.

        Cache entries are immutable and are materialized via hard links before
        pruning, so removing an old entry cannot invalidate a graph already
        handed to ASTRA-Sim.
        """
        if self.max_bytes is None or self.max_bytes <= 0:
            return
        entries = []
        total = 0
        try:
            names = os.listdir(self.cache_dir)
        except FileNotFoundError:
            return
        for name in names:
            if name.startswith("."):
                continue
            entry = os.path.join(self.cache_dir, name)
            manifest_path = self._manifest_path(entry)
            if not os.path.isdir(entry) or not os.path.exists(manifest_path):
                continue
            size = 0
            for root, _, files in os.walk(entry):
                size += sum(os.path.getsize(os.path.join(root, file)) for file in files)
            total += size
            entries.append((os.path.getmtime(entry), entry, size))
        for _, entry, size in sorted(entries):
            if total <= self.max_bytes:
                break
            if os.path.abspath(entry) == os.path.abspath(protected_entry):
                continue
            shutil.rmtree(entry, ignore_errors=True)
            total -= size

    def provide(self, trace_bytes, output_prefix, num_npus, npu_offset=0,
                local_offloading=False):
        """Convert/cache ``trace_bytes`` and return whether the cache hit."""
        if not self.enable_cache:
            trace_dir = os.path.dirname(output_prefix)
            os.makedirs(trace_dir, exist_ok=True)
            trace_path = f"{output_prefix}.trace.txt"
            with open(trace_path, "wb") as f:
                f.write(trace_bytes)
            if self.telemetry is not None:
                with self.telemetry.phase("chakra_converter"):
                    if self.converter_mode == "subprocess":
                        self._convert_subprocess(trace_path, output_prefix, num_npus, npu_offset, local_offloading)
                    else:
                        self._convert_inprocess(trace_path, output_prefix, num_npus, npu_offset, local_offloading)
            elif self.converter_mode == "subprocess":
                self._convert_subprocess(trace_path, output_prefix, num_npus, npu_offset, local_offloading)
            else:
                self._convert_inprocess(trace_path, output_prefix, num_npus, npu_offset, local_offloading)
            try:
                os.remove(trace_path)
            except FileNotFoundError:
                pass
            return False

        key, identity = self._key(trace_bytes, num_npus, npu_offset, local_offloading)
        entry = os.path.join(self.cache_dir, key)
        manifest = self._valid_entry(entry, key)
        hit = manifest is not None
        if not hit:
            if self.telemetry is not None:
                with self.telemetry.phase("chakra_converter"):
                    self._build_entry(entry, trace_bytes, num_npus, npu_offset, local_offloading, key, identity)
            else:
                self._build_entry(entry, trace_bytes, num_npus, npu_offset, local_offloading, key, identity)
            manifest = self._valid_entry(entry, key)
            if manifest is None:
                raise RuntimeError(f"Graph cache entry failed validation: {entry}")
        if self.telemetry is not None:
            with self.telemetry.phase("et_materialization"):
                self._materialize(entry, output_prefix, manifest)
        else:
            self._materialize(entry, output_prefix, manifest)
        try:
            os.utime(entry, None)
        except FileNotFoundError:
            pass
        if not hit and self.max_bytes is not None and self.max_bytes > 0:
            self._entries_since_prune += 1
            if self._entries_since_prune >= _PRUNE_INTERVAL:
                self._prune(entry)
                self._entries_since_prune = 0
        return hit
