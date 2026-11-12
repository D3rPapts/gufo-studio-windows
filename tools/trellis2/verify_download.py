#!/usr/bin/env python3
"""Verifies an HF snapshot against the HF API file tree: size, safetensors header
sanity and SHA-256 (LFS files) or git blob SHA-1 (small files).

    curl -sSL "https://huggingface.co/api/models/<repo>/tree/main?recursive=true" -o tree.json
    python -I verify_download.py <snapshot_dir> tree.json [--delete-bad]

Background: the Xet downloader produced zero-filled safetensors twice on 2026-11-12
(correct size, zero header). hf download does not re-fetch such files, so bad files
are deleted (with --delete-bad, incl. their .cache metadata) and hf download is rerun.
Standard library only; exit code 1 when any file is bad or missing.
"""

import hashlib
import json
import os
import struct
import sys


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path):
    size = os.path.getsize(path)
    h = hashlib.sha1()
    h.update(f"blob {size}\0".encode())
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safetensors_header_ok(path):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            return "too short"
        n = struct.unpack("<Q", raw)[0]
        if n == 0:
            return "header length 0 (zero-filled)"
        if n > 64 * 1024 * 1024 or 8 + n > size:
            return f"implausible header length {n}"
        try:
            header = json.loads(f.read(n).decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            return f"header not JSON: {e}"
    header.pop("__metadata__", None)
    end = max((v["data_offsets"][1] for v in header.values()), default=0)
    if 8 + n + end != size:
        return f"payload ends at {8 + n + end}, file has {size}"
    return f"ok ({len(header)} tensors)"


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    root, tree_path = sys.argv[1], sys.argv[2]
    delete_bad = "--delete-bad" in sys.argv
    with open(tree_path, "r", encoding="utf-8") as f:
        tree = json.load(f)
    bad = 0
    for entry in tree:
        if entry.get("type") != "file":
            continue
        rel = entry["path"]
        if rel == ".gitattributes":
            continue
        path = os.path.join(root, rel.replace("/", os.sep))
        if not os.path.isfile(path):
            print(f"MISSING  {rel}")
            bad += 1
            continue
        size_ok = os.path.getsize(path) == entry.get("size", -1)
        lfs = entry.get("lfs")
        if lfs:
            digest = sha256_file(path)
            expected = lfs.get("oid", "")
            hash_ok = digest == expected
            header = safetensors_header_ok(path) if rel.endswith(".safetensors") else "-"
            status = "OK " if size_ok and hash_ok and (header.startswith("ok") or header == "-") else "BAD"
            print(f"{status}  {rel}  size {'ok' if size_ok else 'MISMATCH'}  sha256 {digest[:16]} "
                  f"{'=' if hash_ok else '!='} {expected[:16]}  header {header}")
        else:
            digest = git_blob_sha1(path)
            expected = entry.get("oid", "")
            hash_ok = digest == expected
            status = "OK " if size_ok and hash_ok else "BAD"
            print(f"{status}  {rel}  size {'ok' if size_ok else 'MISMATCH'}  blob-sha1 {digest[:16]} "
                  f"{'=' if hash_ok else '!='} {expected[:16]}")
        if status == "BAD":
            bad += 1
            if delete_bad:
                os.remove(path)
                meta_dir = os.path.join(root, ".cache", "huggingface", "download", os.path.dirname(rel.replace("/", os.sep)))
                for suffix in (".metadata", ".lock"):
                    meta = os.path.join(meta_dir, os.path.basename(rel) + suffix)
                    if os.path.isfile(meta):
                        os.remove(meta)
                print(f"         deleted {rel} (+ .cache metadata); rerun hf download")
    print(f"{bad} bad/missing file(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
