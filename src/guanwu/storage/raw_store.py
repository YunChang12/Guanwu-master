from __future__ import annotations
import hashlib
import shutil
import logging
from pathlib import Path

logger = logging.getLogger("guanwu")

class RawStore:
    def __init__(self, raw_root: str):
        self.root = Path(raw_root)
        self.root.mkdir(parents=True, exist_ok=True)

    def dataset_dir(self, dataset_id: str) -> Path:
        d = self.root / dataset_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def store_file(self, dataset_id: str, source_path: Path, relative_dest: str, strategy: str = "copy") -> Path:
        """Store a raw file. strategy: 'copy', 'symlink', 'hardlink'."""
        dest = self.dataset_dir(dataset_id) / relative_dest
        if dest.exists():
            logger.debug(f"Raw file already exists: {dest}")
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        if strategy == "symlink":
            dest.symlink_to(source_path.resolve())
        elif strategy == "hardlink":
            dest.hardlink_to(source_path.resolve())
        else:
            shutil.copy2(source_path, dest)
        return dest

    def link_directory(self, dataset_id: str, source_dir: Path, strategy: str = "symlink") -> Path:
        """Link an entire source directory as the raw store for a dataset."""
        dest = self.root / dataset_id / "source"
        if dest.exists():
            logger.debug(f"Raw directory already linked: {dest}")
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        if strategy == "symlink":
            dest.symlink_to(source_dir.resolve())
        else:
            shutil.copytree(source_dir, dest)
        return dest

    def compute_checksum(self, file_path: Path, algorithm: str = "sha256") -> str:
        """Compute file checksum."""
        h = hashlib.new(algorithm)
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    def verify_checksum(self, file_path: Path, expected: str, algorithm: str = "sha256") -> bool:
        """Verify file checksum matches expected."""
        actual = self.compute_checksum(file_path, algorithm)
        return actual == expected

    def file_exists(self, dataset_id: str, relative_path: str) -> bool:
        return (self.dataset_dir(dataset_id) / relative_path).exists()

    def get_path(self, dataset_id: str, relative_path: str) -> Path:
        return self.dataset_dir(dataset_id) / relative_path
