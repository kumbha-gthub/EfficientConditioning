#!/usr/bin/env python3
"""Download the YAHPO Gym surrogate files used by exp23 and exp24.

For each scenario: model.onnx and encoding.json, which the experiments use,
and config_space.json, which yahpo_gym needs for exp24_check_yahpo.py.

The files are taken from https://github.com/slds-lmu/yahpo_data at a fixed
commit (Apache License 2.0) and checked against their SHA-256 hashes.
They are stored in data/<scenario>/ next to this script.

Usage:  python yahpo_data.py
"""

import hashlib
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
COMMIT = "efdab9072f63bd680396cd4b78b927c4a0caaad3"
URL = "https://raw.githubusercontent.com/slds-lmu/yahpo_data/{commit}/{path}"

SHA256 = {
    "lcbench/encoding.json": "d8af928f412bbeb07e58ecd69877fcab35fbc595333d1c1c3f76120d2f2508da",
    "lcbench/model.onnx": "b45c0d85ae39abf8820c8bea5ac82f415763afcbaf3b725e36af452b09cb6dcd",
    "lcbench/config_space.json": "4d1c78a91429b4a37628e13f286cf1abe15f778a09722a4c01714f2d72a8200d",
    "rbv2_xgboost/encoding.json": "0bd3cf9a4c9459a3148adf740f20b347fd11a6714f89348f41467a8f1206172a",
    "rbv2_xgboost/model.onnx": "85125b6d97f553d10cd50f9e632e1a3f1ad44bcb00300dd2880d70dfd043d9f3",
    "rbv2_xgboost/config_space.json": "acf4485d314d86801f61b615d64ffea7859c414c363da9e7f943621468ebbda4",
    "rbv2_ranger/encoding.json": "9f0e5947f6a30ea6daa6e194eed3026117db7d216e4a3df2d36653792ac5f489",
    "rbv2_ranger/model.onnx": "2bf0fc3d5941dc1fa46e43812261f43a3ff66b386669167f434de46fe4093239",
    "rbv2_ranger/config_space.json": "ce36c2f5dd3f01f8ae4b5e58eba1c54c3fafe274d31f17798c87bad780c61521",
    "rbv2_glmnet/encoding.json": "2c060aa8eb6ccc572eadc405bedd8882edec08d4bd35c8f097e103cc5fe5b325",
    "rbv2_glmnet/model.onnx": "21a6280abaa5409e3d4343de7cb6c0d26e192db000091ce52082578f8df094c4",
    "rbv2_glmnet/config_space.json": "6f2a9d7cddac5e7bac17b12870abdc35bc4ebbb245b1a3ce97d2d5fb475ace0b",
    "rbv2_rpart/encoding.json": "e978cd3de5b8fbad4f7dbb7a15bed94738d37c24a0bfb3cbdf33509ba91df49c",
    "rbv2_rpart/model.onnx": "af34bb810107fd809bdd74be227685fb2a1c5cb7b1f3c2dde5629e42ef6819f3",
    "rbv2_rpart/config_space.json": "86de5b9c8f9bc2f82eaafa099b9920445ce6edbe56509c9736af71d53ef4a630",
}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure(scenarios=None, verbose=True):
    """Make sure the files of the given scenarios (default: all) are present
    and unchanged; download missing or modified files.  Returns True if all
    files are in place."""
    ok = True
    for rel, sha in SHA256.items():
        if scenarios is not None and rel.split("/")[0] not in scenarios:
            continue
        path = os.path.join(DATA, *rel.split("/"))
        if os.path.exists(path) and _sha256(path) == sha:
            continue
        url = URL.format(commit=COMMIT, path=rel)
        if verbose:
            print(f"  downloading {rel}")
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".part"
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(tmp, path)
        except Exception as e:                                      # noqa: BLE001
            print(f"  could not download {url}: {e}")
        if not (os.path.exists(path) and _sha256(path) == sha):
            print(f"  missing or wrong file: {path}")
            print(f"  download it manually from {url}")
            ok = False
    return ok


if __name__ == "__main__":
    good = ensure()
    print("all files present" if good else "some files are missing")
    sys.exit(0 if good else 1)
