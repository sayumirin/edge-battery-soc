"""Verify pinned mirror files against the official Panasonic archive and capacity inventory."""

import hashlib
import json
import zipfile
from pathlib import Path


def main():
    archive = Path("data/raw/panasonic-official.zip")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != "0098dc90cfb645beb7a1daff1d9721f7eebfa85dca272e5f6adb6e6ca46c14d8":
        raise ValueError("Official Panasonic archive checksum mismatch")
    manifest = json.loads(Path("data/splits/manifest.json").read_text())["files"]
    out = Path("results/diagnostics-v1")
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        verified = []
        for e in manifest:
            (name,) = [n for n in names if n.endswith(e["path"])]
            actual = hashlib.sha256(z.read(name)).hexdigest()
            if actual != e["sha256"]:
                raise ValueError("Mirror differs from official file: " + name)
            verified.append(dict(file_id=e["file_id"], sha256=actual))
        c20 = [n for n in names if "C20" in Path(n).name and n.endswith(".mat")]
        (readme,) = [n for n in names if "Readme" in n and n.endswith(".txt")]
        text = (
            "\n".join(
                line.rstrip() for line in z.read(readme).decode("cp1252").replace("\r", "").splitlines()
            ).rstrip()
            + "\n"
        )
        (out / "official_dataset_readme.txt").write_text(text, encoding="utf-8", newline="\n")
        result = dict(
            source="https://data.mendeley.com/datasets/wykht8y7tg/1",
            archive_sha256=digest,
            verified_mirror_files=verified,
            c20_files=c20,
            conclusion="Only steps 3-5 repeat at five temperatures. Step 2 C/20 does not. "
            "The archive contains one 25 C C/20 file. Per-temperature C/20 truth cannot be extracted. "
            "Use nominal 2.9 Ah primary labels and explicitly reported 25 C sensitivity; "
            "do not substitute drive endpoints or fabricate five capacities.",
        )
        (out / "official_archive_audit.json").write_text(json.dumps(result, indent=2))
        print(f"Official archive verified: {len(verified)} matching files, {len(c20)} C/20 reference")


if __name__ == "__main__":
    main()
