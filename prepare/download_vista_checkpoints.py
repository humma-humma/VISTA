"""Download and verify the released VISTA checkpoints.

Files, URLs and SHA256 checksums are listed in `prepare/vista_checkpoints.json`. Each file is
placed at the path the scripts expect (relative to --root), e.g.

  checkpoints/t2m/MARDM-DDPM-XL/model/final_diffmlp_diff_v4_cfg_cross_batch_hybrid_ema_fix.tar
  checkpoints/100styles/DAE/final_finetune_diffmlp_decoder_fixed_hybrid_ema_fix.tar
  checkpoints/style_classifier/style_classifier_final.pt

Downloads go to a `.part` file and are renamed only after the checksum matches. Existing files
are verified and kept; a file with a different checksum is never overwritten without --force.
Google Drive links are fetched with gdown, everything else (Hugging Face, Zenodo, plain HTTPS)
with a streaming HTTP download.

Usage (from the repository root):
    python prepare/download_vista_checkpoints.py                 # all required checkpoints
    python prepare/download_vista_checkpoints.py --all           # also optional ones
    python prepare/download_vista_checkpoints.py --only mardm dae
    python prepare/download_vista_checkpoints.py --verify-only   # check local files, no download
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY = os.path.join(HERE, "vista_checkpoints.json")


def sha256sum(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def http_download(url, dst):
    req = urllib.request.Request(url, headers={"User-Agent": "vista-downloader"})
    with urllib.request.urlopen(req) as r, open(dst, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            b = r.read(1 << 22)
            if not b:
                break
            f.write(b)
            done += len(b)
            if total:
                print(f"\r  {done / 2**20:9.1f} / {total / 2**20:9.1f} MiB", end="", flush=True)
    print()


def download(url, dst):
    if "drive.google.com" in url:
        subprocess.run(["gdown", "--fuzzy", url, "-O", dst], check=True)
    else:
        http_download(url, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="repository root (default: current directory)")
    ap.add_argument("--only", nargs="+", default=None, help="subset of registry keys")
    ap.add_argument("--all", action="store_true", help="include optional checkpoints")
    ap.add_argument("--verify-only", action="store_true", help="only verify files that are already present")
    ap.add_argument("--force", action="store_true", help="replace local files whose checksum does not match")
    args = ap.parse_args()

    with open(REGISTRY, encoding="utf-8") as f:
        registry = json.load(f)["checkpoints"]
    keys = args.only or [k for k, v in registry.items() if args.all or not v.get("optional", False)]
    unknown = [k for k in keys if k not in registry]
    if unknown:
        sys.exit(f"unknown checkpoint key(s): {unknown}; available: {list(registry)}")

    failed = []
    for key in keys:
        e = registry[key]
        dst = os.path.join(args.root, e["path"])
        print(f"[{key}] {e['path']}")
        if os.path.exists(dst):
            digest = sha256sum(dst)
            if digest == e["sha256"]:
                print("  present, checksum OK")
                continue
            print(f"  present but checksum differs ({digest[:12]}... != {e['sha256'][:12]}...)")
            if not args.force or args.verify_only:
                failed.append(key)
                continue
        elif args.verify_only:
            print("  missing")
            failed.append(key)
            continue
        if not e.get("url"):
            print("  no download URL configured in prepare/vista_checkpoints.json")
            failed.append(key)
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        part = dst + ".part"
        try:
            download(e["url"], part)
            digest = sha256sum(part)
            if digest != e["sha256"]:
                raise RuntimeError(f"checksum mismatch after download ({digest[:12]}...)")
            if os.path.exists(dst):
                os.remove(dst)  # only reached with --force
            shutil.move(part, dst)
            print("  downloaded, checksum OK")
        except Exception as ex:
            print(f"  FAILED: {ex}")
            if os.path.exists(part):
                os.remove(part)
            failed.append(key)

    if failed:
        sys.exit(f"\n{len(failed)} checkpoint(s) not ready: {failed}")
    print("\nAll requested checkpoints are present and verified.")


if __name__ == "__main__":
    main()
