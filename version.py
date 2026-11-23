import hashlib
import json
import os
import sys

VERSION = "0.4.1"

PIPELINE_FILES = [
    "facial_analysis.py",
    "face_dynamics.py",
    "feature_extractor.py",
    "scoring.py",
    "fusion.py",
    "routing.py",
]


def _here():
    return os.path.dirname(os.path.abspath(__file__))


def source_fingerprint():
    base = _here()
    h = hashlib.sha256()
    found = False
    for name in PIPELINE_FILES:
        path = os.path.join(base, name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = f.read().replace(b"\r\n", b"\n")
            h.update(name.encode())
            h.update(data)
            found = True
    if found:
        return h.hexdigest()[:12]
    info_path = os.path.join(getattr(sys, "_MEIPASS", base), "build_info.json")
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            return json.load(f)["fingerprint"]
    except (OSError, KeyError, ValueError):
        return "unknown"


def info():
    return {
        "version": VERSION,
        "fingerprint": source_fingerprint(),
        "frozen": bool(getattr(sys, "frozen", False)),
    }


def write_build_info(path="build_info.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"version": VERSION, "fingerprint": source_fingerprint()}, f)
    return path


if __name__ == "__main__":
    if "--write" in sys.argv:
        print("wrote", write_build_info())
    else:
        print(json.dumps(info()))
