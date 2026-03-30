from __future__ import annotations

from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config


def upload_directory(
    *,
    local_dir: str | Path,
    bucket: str,
    prefix: str,
    region: str | None = None,
) -> list[str]:
    base = Path(local_dir).resolve()
    if not base.exists() or not base.is_dir():
        raise ValueError(f"Directory not found: {base}")

    session = boto3.session.Session(region_name=region)
    s3 = session.client(
        "s3",
        config=Config(
            retries={"max_attempts": 10, "mode": "standard"},
            connect_timeout=10,
            read_timeout=120,
        ),
    )
    transfer_config = TransferConfig(
        multipart_threshold=64 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        max_concurrency=8,
        use_threads=True,
    )

    normalized_prefix = prefix.strip("/")
    uploaded_keys: list[str] = []
    files = sorted(path for path in base.rglob("*") if path.is_file())
    for path in files:
        rel = path.relative_to(base).as_posix()
        key = f"{normalized_prefix}/{rel}" if normalized_prefix else rel
        s3.upload_file(
            str(path),
            bucket,
            key,
            Config=transfer_config,
        )
        uploaded_keys.append(key)

    return uploaded_keys
