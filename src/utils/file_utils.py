import hashlib
from typing import Union
from pathlib import Path

def calculate_file_hash(file_content: Union[bytes, Path, str]) -> str:
    """
    Calculates the SHA-256 hash of a file or byte content.
    Useful for deduplication.

    Args:
        file_content: Path to the file as a string or Path object, or raw bytes.

    Returns:
        Hex string of the SHA-256 hash.
    """
    sha256_hash = hashlib.sha256()

    if isinstance(file_content, (str, Path)):
        path = Path(file_content)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
    elif isinstance(file_content, bytes):
        sha256_hash.update(file_content)
    else:
        raise ValueError("file_content must be bytes or a valid file path")

    return sha256_hash.hexdigest()
