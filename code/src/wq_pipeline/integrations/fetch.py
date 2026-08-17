from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

import requests


def download_file(url: str, output_path: Path, params: dict[str, str] | None = None, timeout: int = 120) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_url = f"{url}?{urlencode(params)}" if params else url

    with requests.get(final_url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with output_path.open("wb") as fout:
            for chunk in response.iter_content(chunk_size=1_048_576):
                if chunk:
                    fout.write(chunk)
    return output_path


def build_eea_download_url(base_url: str, dataset_id: str, resource_path: str) -> str:
    root = base_url.rstrip("/")
    return f"{root}/{dataset_id}/{resource_path.lstrip('/')}"
