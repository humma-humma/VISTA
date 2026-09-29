"""Download and verify the released VISTA checkpoints, evaluators and GloVe vectors.

Files, Google Drive links and SHA256 checksums are listed in `prepare/vista_checkpoints.json`.
Each file is placed at the path the scripts expect (relative to --root), e.g.

  checkpoints/t2m/MARDM-DDPM-XL/model/final_diffmlp_diff_v4_cfg_cross_batch_hybrid_ema_fix.tar
  checkpoints/100styles/DAE/final_finetune_diffmlp_decoder_fixed_hybrid_ema_fix.tar
  checkpoints/style_classifier/style_classifier_final.pt
  checkpoints/t2m/text_mot_match{,_clip}/model/finest.tar   (+ copies under checkpoints/100styles/)
  glove/our_vab_{data.npy,idx.pkl,words.pkl}

Downloads go to a `.part` file and are used only after the checksum matches. Existing files are
verified and kept; a file with a different checksum is never overwritten without --force.
Archives (e.g. glove.zip) are verified, extracted and removed. Google Drive links are fetched with
gdown, anything else with a streaming HTTPS download.

Usage (from the repository root):
    python prepare/download_vista_checkpoints.py                 # everything needed for sampling + evaluation
    python prepare/download_vista_checkpoints.py --all           # also Stage-1 DualAE and the refiner
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
import zipfile

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
        subprocess.run([sys.executable, "-m", "gdown", "--fuzzy", url, "-O", dst], check=True)
    else:
        http_download(url, dst)


def fetch(e, dst):
    """Download e['url'] to dst via a verified .part file."""
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    part = dst + ".part"
    try:
        download(e["url"], part)
        digest = sha256sum(part)
        if digest != e["sha256"]:
            raise RuntimeError(f"checksum mismatch after download ({digest[:12]}...)")
        if os.path.exists(dst):
            os.remove(dst)  # only reached with --force
        shutil.move(part, dst)
    finally:
        if os.path.exists(part):
            os.remove(part)


def handle_archive(key, e, root, args):
    """Zip archive: 'provides' lists the extracted files that mark it as present."""
    provided = [os.path.join(root, p) for p in e["provides"]]
    if all(os.path.exists(p) for p in provided):
        print("  present")
        return True
    if args.verify_only:
        print("  missing: " + ", ".join(p for p, full in zip(e["provides"], provided) if not os.path.exists(full)))
        return False
    archive = os.path.join(root, e["path"])
    fetch(e, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(os.path.join(root, e.get("extract_to", ".")))
    os.remove(archive)
    ok = all(os.path.exists(p) for p in provided)
    print("  downloaded, checksum OK, extracted" if ok else "  extracted, but expected files are missing")
    return ok


def handle_file(key, e, root, args):
    dst = os.path.join(root, e["path"])
    if os.path.exists(dst):
        digest = sha256sum(dst)
        if digest == e["sha256"]:
            print("  present, checksum OK")
        else:
            print(f"  present but checksum differs ({digest[:12]}... != {e['sha256'][:12]}...)")
            if args.verify_only or not args.force:
                return False
            fetch(e, dst)
            print("  re-downloaded, checksum OK")
    elif args.verify_only:
        print("  missing")
        return False
    else:
        fetch(e, dst)
        print("  downloaded, checksum OK")
    # additional locations that need the same file (e.g. evaluators under checkpoints/100styles/)
    for extra in e.get("copies", []):
        extra_dst = os.path.join(root, extra)
        if os.path.exists(extra_dst) and sha256sum(extra_dst) == e["sha256"]:
            continue
        if args.verify_only:
            print(f"  copy missing: {extra}")
            return False
        os.makedirs(os.path.dirname(extra_dst), exist_ok=True)
        shutil.copy2(dst, extra_dst)
        print(f"  copied to {extra}")
    return True


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
        print(f"[{key}] {e['path']}")
        if not e.get("url") and not args.verify_only:
            print("  no download URL configured in prepare/vista_checkpoints.json")
            failed.append(key)
            continue
        try:
            ok = (handle_archive if e.get("archive") else handle_file)(key, e, args.root, args)
        except Exception as ex:
            print(f"  FAILED: {ex}")
            ok = False
        if not ok:
            failed.append(key)

    if failed:
        sys.exit(f"\n{len(failed)} item(s) not ready: {failed}")
    print("\nAll requested files are present and verified.")


if __name__ == "__main__":
    main()
