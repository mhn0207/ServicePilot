"""Capture explicit source files and installed versions, never dotenv or user data."""
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import zipfile

SOURCE_DIRS = ('agents', 'api', 'core', 'evaluation', 'memory', 'monitor', 'scripts', 'tooling')


def source_identity(root):
    root = Path(root)
    paths = [p for name in SOURCE_DIRS for p in (root/name).rglob('*.py')
             if not p.is_symlink() and not any(parent.is_symlink() for parent in p.parents)]
    paths += [root/name for name in ('requirements.txt', 'requirements-dev.txt') if (root/name).is_file()]
    files = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(paths)}
    manifest = json.dumps(files, sort_keys=True, separators=(',', ':'))
    versions = {}
    for name in ('pydantic', 'anthropic', 'httpx', 'chromadb', 'onnxruntime', 'tokenizers', 'fastapi'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return dict(source_hash=hashlib.sha256(manifest.encode()).hexdigest(), source_manifest=manifest,
                python=platform.python_version(), packages=json.dumps(versions, sort_keys=True))


def write_source_bundle(root, output, identity):
    """Verify captured bytes before archiving; fail rather than claim another version."""
    if 'source_manifest' not in identity:
        return  # Historical reports and injected test identities have no bundle.
    root, output = Path(root), Path(output)
    files = json.loads(identity['source_manifest'])
    contents = {}
    for name, digest in files.items():
        path = root/name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('source changed or escaped root')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError('source changed during evaluation')
        contents[name] = data
    with zipfile.ZipFile(output/'source.zip', 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, data in sorted(contents.items()):
            bundle.writestr(name, data)
