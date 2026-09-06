# -*- coding: utf-8 -*-
"""下载 Pop-K 语料(Zenodo 直链)并校验大小。用法: python scripts/download_dataset.py"""
import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "https://zenodo.org/records/14791511/files/popk_dataset_300k_mid.tar.gz?download=1"
EXPECTED_BYTES = 56168893          # 与 Zenodo 记录一致
DST = Path(__file__).resolve().parent.parent / "data" / "popk_300k_mid.tar.gz"


def main():
    DST.parent.mkdir(parents=True, exist_ok=True)
    if DST.exists() and DST.stat().st_size == EXPECTED_BYTES:
        print("already present:", DST, DST.stat().st_size)
        return
    print("downloading", URL)
    tmp = DST.with_suffix(".part")
    urllib.request.urlretrieve(URL, str(tmp))          # 自动跟随重定向
    if tmp.stat().st_size != EXPECTED_BYTES:
        print("size mismatch", tmp.stat().st_size, "expected", EXPECTED_BYTES, file=sys.stderr)
        sys.exit(1)
    tmp.replace(DST)
    print("ok:", DST)


if __name__ == "__main__":
    main()
