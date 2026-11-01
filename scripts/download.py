"""Download pinned split files, verify Git blob and persist/check SHA-256."""

import hashlib
import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest_path = ROOT / "data/splits/manifest.json"
    m = json.loads(manifest_path.read_text())

    def fetch(e):
        p = ROOT / "data/raw" / e["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            data = p.read_bytes()
        else:
            url = (
                f"https://raw.githubusercontent.com/{m['repository']}/{m['revision']}/"
                + urllib.parse.quote(e["path"])
            )
            with urllib.request.urlopen(url, timeout=120) as r:
                data = r.read()
        blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        digest = hashlib.sha256(data).hexdigest()
        if blob != e["git_blob"] or ("sha256" in e and digest != e["sha256"]):
            raise ValueError(f"Checksum mismatch: {p}")
        if not p.exists():
            tmp = p.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(p)
        e["sha256"] = digest
        print("Verified", p.name, flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, m["files"]))
    manifest_path.write_text(json.dumps(m, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
