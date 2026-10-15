"""Pinned, checksum-verified official model downloads and component discovery."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
PLUS_REPO = "ruchangyao/FlashVSR_plus"
PLUS_REV = "46b79520946ac67dec94d1566490690953c7debc"
V1_REPO = "JunhaoZhuang/FlashVSR-v1.1"
V1_REV = "27561b186ded3402d7c975f4fd722e2885b6135f"
# role, local path, repository, revision, remote path, bytes, LFS SHA256
FILES = [
    ("turbo_dit", "turbo/dit.safetensors", PLUS_REPO, PLUS_REV, "turbo/dit.safetensors", 5676070392, "7862b92178176fa3aebc34c6c071daca6b56ab1dddb8e7ff9a01c47f5226ec8b"),
    ("turbo_lq", "turbo/lq_proj_in.safetensors", PLUS_REPO, PLUS_REV, "turbo/lq_proj_in.safetensors", 1151384272, "e77b378ae12342eefbb58f924256f362a151b93ce73dd32808fcdaca9c051496"),
    ("base_dit", "base/dit.safetensors", PLUS_REPO, PLUS_REV, "base/dit.safetensors", 5676070392, "58d64c1f1b3a559bafb69bb014f3ab2745e800c89435afdc154966eb96bc7459"),
    ("base_lq", "base/lq_proj_in.safetensors", PLUS_REPO, PLUS_REV, "base/lq_proj_in.safetensors", 1151384272, "a1fe997f19060171db8dd4d58c30debc78ddfcc357b63481798792a38a356d8d"),
    ("tiny", "TCDecoder.ckpt", PLUS_REPO, PLUS_REV, "TCDecoder.ckpt", 189018333, "e224bdcf2f52745cbf4d393ff5374c2ba09e90285d5d19062d2bf63b915b6161"),
    ("vae", "Wan2.1_VAE.pth", PLUS_REPO, PLUS_REV, "Wan2.1_VAE.pth", 507609880, "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981"),
    ("v1_dit", "v1/dit.safetensors", V1_REPO, V1_REV, "diffusion_pytorch_model_streaming_dmd.safetensors", 5676070392, "bd28180edcf3446c028e32fc6b731a80bf7e4da2ab4caac3186b9499964d37be"),
    ("v1_lq", "v1/LQ_proj_in.ckpt", V1_REPO, V1_REV, "LQ_proj_in.ckpt", 575694948, "d6d011cdaaba6a52645086caa08fa04124e746f6ca568140a24007591142bfd2"),
]
# The release mirror preserves the exact official bytes and checkpoint names.
MIRROR_REPO = "t8star/Flashvsr-Plus-Comfy"
MIRROR_REV = "56c806f1357d90c8a0b4b054e21bcefe3cb9ac1d"
UPSTREAM_FILES = FILES
FILES = [(role, local, MIRROR_REPO, MIRROR_REV, local, size, digest)
         for role, local, repo, revision, remote, size, digest in UPSTREAM_FILES]
INT8_FILES = [
    ("turbo_int8_convrot", "turbo/dit_int8_convrot.safetensors", MIRROR_REPO, MIRROR_REV,
     "turbo/dit_int8_convrot.safetensors", 1448886272, "14150f06a5661e6da479f7583766c55e310e16cf3ce538ba56dc70188f03d773"),
    ("base_int8_convrot", "base/dit_int8_convrot.safetensors", MIRROR_REPO, MIRROR_REV,
     "base/dit_int8_convrot.safetensors", 1448886272, "beaa478bf63717df2a73e472e1a1383a2731b8f6556fa29afd95410249256732"),
    ("v1_int8_convrot", "v1/dit_int8_convrot.safetensors", MIRROR_REPO, MIRROR_REV,
     "v1/dit_int8_convrot.safetensors", 1448886264, "4b0bb12bd5e6466b4432b95d98e10e55f4579a5de5a4f0276fae725369ff4317"),
]
_DOWNLOAD_LOCK = threading.RLock()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(path: Path, size: int, digest: str) -> None:
    if not path.is_file() or path.stat().st_size != size or sha256(path) != digest:
        raise ValueError(f"Model checksum/size mismatch: {path}. Preserve the file and download a clean official copy.")


def verified_model(path: Path, size: int, digest: str) -> None:
    """Avoid rehashing unchanged official files on every workflow execution."""
    stamp = path.stat()
    receipt = path.with_name(path.name + ".verified.json")
    expected = {"size": size, "mtime_ns": stamp.st_mtime_ns, "sha256": digest}
    try:
        if stamp.st_size == size and json.loads(receipt.read_text(encoding="utf-8")) == expected:
            return
    except (OSError, ValueError):
        pass
    verify(path, size, digest)
    try:
        receipt.write_text(json.dumps(expected), encoding="utf-8")
    except OSError:
        pass


def download_file(entry: tuple, directory: Path, progress: Callable[[str], None] | None = None) -> Path:
    import requests
    role, local, repo, revision, remote, size, digest = entry
    target = directory / local
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        verify(target, size, digest)
        return target
    part = target.with_name(target.name + ".download")
    ledger = target.with_name(target.name + ".download.json")
    chunk = 32*1024*1024
    done = set()
    if ledger.exists() and part.exists() and part.stat().st_size == size:
        try:
            saved = json.loads(ledger.read_text())
            if isinstance(saved, dict) and saved.get("sha256") == digest:
                done = {i for i in saved.get("chunks", [])
                        if isinstance(i, int) and 0 <= i < (size+chunk-1)//chunk}
        except (OSError, ValueError, TypeError):
            done = set()
    if not part.exists() or part.stat().st_size != size:
        with part.open("wb") as f:
            f.truncate(size)
    lock = threading.Lock()
    cancelled = threading.Event()
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{remote}"

    def fetch(i):
        start, end = i*chunk, min((i+1)*chunk, size)-1
        for attempt in range(4):
            if cancelled.is_set():
                raise InterruptedError("Model download cancelled")
            try:
                with requests.get(url, headers={"Range": f"bytes={start}-{end}"},
                                  stream=True, timeout=(15, 20)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise RuntimeError("Official model server did not honor the requested byte range")
                    count = 0
                    with part.open("r+b") as f:
                        f.seek(start)
                        for data in response.iter_content(2*1024*1024):
                            if cancelled.is_set():
                                raise InterruptedError("Model download cancelled")
                            count += len(data)
                            if count > end-start+1:
                                raise ValueError("Download range exceeded expected size")
                            f.write(data)
                    if count != end-start+1:
                        raise ValueError("Incomplete model range")
                with lock:
                    done.add(i)
                    ledger.write_text(json.dumps({"sha256": digest, "chunks": sorted(done)}))
                return
            except (requests.RequestException, RuntimeError, ValueError):
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)

    jobs = [i for i in range((size+chunk-1)//chunk) if i not in done]
    with ThreadPoolExecutor(max_workers=8) as pool:
        pending = [pool.submit(fetch, i) for i in jobs]
        try:
            for future in as_completed(pending):
                future.result()
                if progress:
                    progress(f"Downloading {role}: {min(len(done)*chunk, size)/size:.0%}")
        except BaseException:
            cancelled.set()
            for future in pending:
                future.cancel()
            raise
    try:
        verify(part, size, digest)
    except ValueError:
        ledger.unlink(missing_ok=True)
        raise
    os.replace(part, target)
    ledger.unlink(missing_ok=True)
    return target


def model_paths(mode: str, directory: str | Path, decoder: str, *, auto_download=True,
                progress: Callable[[str], None] | None = None, include_dit=True, include_int8=False):
    from .inference.loading import ModelPaths
    if mode not in ("turbo", "base", "v1") or decoder not in ("tiny", "vae"):
        raise ValueError("Unknown official model mode or decoder")
    directory = Path(directory).expanduser().resolve()
    wanted = {mode+"_lq", decoder}
    if include_dit:
        wanted.add(mode+"_dit")
    if include_int8:
        wanted.add(mode+"_int8_convrot")
    found = {}
    with _DOWNLOAD_LOCK:
        for entry in (*FILES, *INT8_FILES):
            role, local, _, _, _, size, digest = entry
            if role not in wanted:
                continue
            path = directory/local
            if not path.exists():
                if not auto_download:
                    raise FileNotFoundError(f"Missing official model: {path}; enable auto_download or run the download CLI")
                path = download_file(entry, directory, progress)
            if path.stat().st_size != size:
                raise ValueError(f"Model file is incomplete: {path}")
            verified_model(path, size, digest)
            found[role] = str(path)
    context = ROOT/"prompt_tensor/posi_prompt.pth"
    if not context.is_file():
        context = Path(__file__).resolve().parent/"assets/posi_prompt.pth"
    if not context.is_file():
        raise FileNotFoundError(f"Missing bundled prompt tensor: {context}")
    return ModelPaths(dit=found.get(mode+"_int8_convrot") or found.get(mode+"_dit"), lq_projection=found[mode+"_lq"], context=str(context),
                      tiny_decoder=found.get("tiny"), vae=found.get("vae"))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT/"ckpts")
    parser.add_argument("--modes", nargs="+", choices=["turbo", "base", "v1"], default=["turbo", "base"])
    parser.add_argument("--vae", action="store_true")
    args = parser.parse_args()
    for mode in args.modes:
        model_paths(mode, args.model_dir, "tiny", progress=print)
        if args.vae:
            model_paths(mode, args.model_dir, "vae", progress=print)


if __name__ == "__main__":
    main()
