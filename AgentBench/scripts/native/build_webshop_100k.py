#!/usr/bin/env python3
"""Build the exact WebShop 100k text assets without Docker.

Run this with ``.native/webshop-venv/bin/python`` after preparing the patched
WebShop source at ``.native/webshop-src`` and placing the three official JSON
files under its ``data`` directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(os.environ.get("WEBSHOP_RUNTIME_SOURCE", ROOT / ".native/webshop-src")).resolve()
DATA = SOURCE / "data"
SEARCH = SOURCE / "search_engine"

EXPECTED = {
    "items_shuffle.json": "2ef591d65df3af89e972ab72468eb82cbf124d876552d9f3678667edd620a6c8",
    "items_ins_v2.json": "1d36af476bdb8f82a5da62bd8acdabe54cd8de2fa84010d37da5c4890feb447e",
    "items_human_ins.json": "cf78667548a71786e1d9049c24b802e48e1084ad4bb021cae56ce1f6d96954a3",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def verify_inputs() -> None:
    for name, expected in EXPECTED.items():
        path = DATA / name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = digest(path)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {path}: {actual}")
        print(f"verified {name}: {actual}", flush=True)


def make_document(product: dict) -> dict:
    option_texts = []
    for option_name, option_contents in product.get("options", {}).items():
        option_texts.append(f"{option_name}: {', '.join(option_contents)}")
    option_text = ", and ".join(option_texts)
    contents = " ".join(
        [
            product["Title"],
            product["Description"],
            product["BulletPoints"][0],
            option_text,
        ]
    ).lower()
    return {"id": product["asin"], "contents": contents, "product": product}


def select_products() -> list[dict]:
    sys.path.insert(0, str(SOURCE))
    from web_agent_site.engine import engine

    # Avoid millions of terminal progress updates in unattended native builds.
    engine.tqdm = lambda values, total=None: values
    products, *_ = engine.load_products(filepath=str(DATA / "items_shuffle.json"))
    selected = products[:100_000]
    if len(selected) != 100_000:
        raise RuntimeError(f"expected 100000 valid products, got {len(selected)}")
    return selected


def write_assets(products: list[dict]) -> None:
    product_path = DATA / "items_shuffle_100k.json"
    attribute_path = DATA / "items_ins_v2_100k.json"
    resource_dir = SEARCH / "resources_100k"
    resource_dir.mkdir(parents=True, exist_ok=True)
    document_path = resource_dir / "documents.jsonl"

    with product_path.open("w") as handle:
        json.dump(products, handle)

    asins = {item["asin"] for item in products}
    with (DATA / "items_ins_v2.json").open() as handle:
        attributes = json.load(handle)
    filtered = {key: value for key, value in attributes.items() if key in asins}
    with attribute_path.open("w") as handle:
        json.dump(filtered, handle)

    with document_path.open("w") as handle:
        for product in products:
            handle.write(json.dumps(make_document(product)) + "\n")

    print(
        f"wrote {len(products)} products, {len(filtered)} attribute records, "
        f"and {len(products)} Lucene documents",
        flush=True,
    )


def select_runtime_paths() -> None:
    path = SOURCE / "web_agent_site/utils.py"
    text = path.read_text()
    text = text.replace("../data/items_ins_v2.json'", "../data/items_ins_v2_100k.json'")
    text = text.replace("../data/items_shuffle.json'", "../data/items_shuffle_100k.json'")
    path.write_text(text)


def build_index() -> None:
    index = SEARCH / "indexes_100k"
    if index.exists() and any(index.iterdir()):
        print(f"Lucene index already exists: {index}", flush=True)
    else:
        index.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pyserini.index.lucene",
                "--collection",
                "JsonCollection",
                "--input",
                "resources_100k",
                "--index",
                "indexes_100k",
                "--generator",
                "DefaultLuceneDocumentGenerator",
                "--threads",
                "1",
                "--storePositions",
                "--storeDocvectors",
                "--storeRaw",
            ],
            cwd=SEARCH,
            check=True,
        )

    from pyserini.search.lucene import LuceneSearcher

    indexed = LuceneSearcher(str(index)).num_docs
    if indexed != 99_995:
        raise RuntimeError(f"expected 99995 indexed WebShop documents, got {indexed}")
    print(f"verified Lucene index: {indexed} documents", flush=True)


def main() -> None:
    verify_inputs()
    outputs = [
        DATA / "items_shuffle_100k.json",
        DATA / "items_ins_v2_100k.json",
        SEARCH / "resources_100k/documents.jsonl",
    ]
    if all(path.is_file() for path in outputs):
        print("100k JSON assets already exist", flush=True)
    elif any(path.exists() for path in outputs):
        raise RuntimeError("partial 100k build found; remove its outputs before retrying")
    else:
        write_assets(select_products())
    select_runtime_paths()
    build_index()


if __name__ == "__main__":
    main()
