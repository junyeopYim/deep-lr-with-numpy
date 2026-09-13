"""노트북 하나를 새 커널에서 실행하고 그림·표시·오류를 점검합니다.

사용: python scripts/check_notebook.py notebooks/00_기초/00b_vectorization.ipynb [--out 그림_폴더]

- 실행 결과(출력 포함)를 노트북에 저장합니다.
- --out 폴더에 셀별 PNG를 저장해 눈으로 확인할 수 있게 합니다(기본: figures/check/<노트북 이름>/).
- 마크다운 셀의 표시(🔑/🔍/📎/✏️) 통계, 표시 없는 셀, 제목 앞에 표시가 붙어 제목이 깨지는 셀을 보고합니다.
"""

import argparse
import base64
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import nbformat
from nbclient import NotebookClient

TAGS = ("🔑", "🔍", "📎", "✏️")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    path = args.notebook.resolve()
    out = args.out or (Path(__file__).resolve().parents[1] / "figures" / "check" / path.stem)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("cell*.png"):
        old.unlink()

    nb = nbformat.read(path, as_version=4)
    start = time.time()
    try:
        NotebookClient(nb, timeout=args.timeout, kernel_name="python3",
                       resources={"metadata": {"path": str(path.parent)}}).execute()
    except Exception as error:  # 실패해도 여기까지의 출력을 저장하고 보고합니다
        print("EXEC ERROR:", str(error)[:1500])
    nbformat.write(nb, path)

    images = errors = 0
    stderr = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                errors += 1
                print(f"ERR cell {i}: {output.get('ename')}: {str(output.get('evalue'))[:300]}")
            if output.get("output_type") == "stream" and output.get("name") == "stderr":
                stderr.append((i, "".join(output["text"])[:200]))
            if "image/png" in output.get("data", {}):
                images += 1
                (out / f"cell{i:02d}.png").write_bytes(base64.b64decode(output["data"]["image/png"]))

    tags = Counter()
    untagged, broken_headers = [], []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "markdown" or cell.source.startswith("# "):
            continue
        match = re.match(r"(#+ )?(🔑|🔍|📎|✏️)", cell.source)
        if match:
            tags[match.group(2)] += 1
        else:
            untagged.append(i)
        if re.search(r"^(🔑|🔍|📎|✏️)[^\n]*#{2,3} ", cell.source):
            broken_headers.append(i)
        if re.search(r"(?<!/)~", cell.source):
            print(f"WARN cell {i}: URL 밖의 물결표(취소선 위험)")

    print(f"{path.name}: {time.time() - start:.1f}s · cells {len(nb.cells)} · images {images} · errors {errors} · stderr {stderr}")
    print(f"tags {dict(tags)} · untagged {untagged} · header-in-line {broken_headers} · 그림 폴더 {out}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
