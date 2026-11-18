"""Prepare the versioned Chroma embedding model during a Docker build."""

import hashlib
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path


URL = "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz"
SHA256 = "913d7300ceae3b2dbc2c50d1de4baacab4be7b9380491c27fab7418616a16ec3"
FILENAME = "onnx.tar.gz"


def valid(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == SHA256


def download(path: Path):
    partial = path.with_suffix(".partial")
    deadline = time.monotonic() + 120
    try:
        with urllib.request.urlopen(URL, timeout=15) as response, partial.open("wb") as target:
            while chunk := response.read1(1024 * 1024):
                if time.monotonic() >= deadline:
                    raise TimeoutError("ONNX download exceeded 120 seconds")
                target.write(chunk)
        if not valid(partial):
            raise ValueError("ONNX archive checksum mismatch")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def main():
    provided = Path("/onnx-input") / FILENAME
    cached = Path("/onnx-cache") / FILENAME
    if provided.exists():
        if not valid(provided):
            raise SystemExit("Invalid onnx_input archive: checksum mismatch")
        shutil.copyfile(provided, cached)
    if not valid(cached):
        try:
            download(cached)
        except (OSError, ValueError) as ex:
            raise SystemExit(
                f"Unable to obtain the embedding model ({ex}). "
                "Pass an existing model archive with --build-context onnx_input=PATH."
            ) from ex
    destination = Path.home() / ".cache/chroma/onnx_models/all-MiniLM-L6-v2"
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(cached, "r:gz") as archive:
        archive.extractall(destination, filter="data")


if __name__ == "__main__":
    main()
