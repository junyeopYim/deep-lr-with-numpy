"""노트북 재작성 전후의 출력(print 문장, 그림 바이트)이 같은지 비교합니다. WRITING.md 7절의 "출력이 같은 재작성" 증명용.

사용:
  1) 고치기 전에 스냅샷:  python scripts/compare_notebook_outputs.py snapshot notebooks/00_기초/01_gradients_and_learning.ipynb
     → figures/check/snapshots/<노트북 이름>.outputs.json (gitignore 폴더). --out 으로 다른 경로 지정 가능.
  2) 고치고 check_notebook.py 로 새 커널 실행한 뒤:
     python scripts/compare_notebook_outputs.py compare figures/check/snapshots/<이름>.outputs.json notebooks/…/<이름>.ipynb

비교하는 것: 코드 셀 stream 출력을 순서대로 이은 문장(셀을 나누거나 합쳐도 비교 가능), PNG 그림의 md5, 그 밖의 출력(execute_result 문자열·오류).
시간 측정 수치("12.345 ms", "1,234번 갱신", "0.34초")는 실행마다 달라지므로 <time>으로 가린 뒤 비교합니다.
차이가 하나라도 남으면 종료 코드 1이며, 남은 차이는 모두 보고에서 설명해야 합니다(허용되는 예: 검산만 있던 셀에 추가한 통과 print, 힌트의 관용구 교체).
"""

import argparse
import difflib
import hashlib
import json
import re
import sys
from pathlib import Path

import nbformat

TIME_PATTERN = re.compile(r"\d+\.\d+ ms|\d[\d,]*번 갱신|\d+\.\d+초")


def snapshot(path: Path) -> list[dict]:
    nb = nbformat.read(path, as_version=4)
    cells = []
    for index, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        text, images, others = "", [], []
        for output in cell.get("outputs", []):
            kind = output.get("output_type")
            if kind == "stream":
                text += "".join(output.get("text", ""))
            elif kind in ("display_data", "execute_result"):
                data = output.get("data", {})
                if "image/png" in data:
                    images.append(hashlib.md5(data["image/png"].encode()).hexdigest())
                elif "text/plain" in data:
                    others.append("".join(data["text/plain"]))
            elif kind == "error":
                others.append(f"ERROR {output.get('ename')}: {output.get('evalue')}")
        cells.append({"cell": index, "text": text, "images": images, "others": others})
    return cells


def mask(line: str) -> str:
    return TIME_PATTERN.sub("<time>", line)


def compare(before: list[dict], after: list[dict], name: str) -> int:
    before_lines = "".join(c["text"] for c in before).splitlines()
    after_lines = "".join(c["text"] for c in after).splitlines()
    before_images = [h for c in before for h in c["images"]]
    after_images = [h for c in after for h in c["images"]]
    identical = sum(1 for h in before_images if h in after_images)
    print(f"{name}: 출력 줄 {len(before_lines)} → {len(after_lines)} · 그림 {len(before_images)} → {len(after_images)}"
          f" (바이트 동일 {identical}/{len(before_images)})")

    diff = [line for line in difflib.unified_diff([mask(l) for l in before_lines], [mask(l) for l in after_lines],
                                                  lineterm="", n=0)
            if line[:1] in "+-" and not line.startswith(("+++", "---"))]
    problems = 0
    if diff:
        problems += len(diff)
        print("  출력 문장 차이 (시간 수치는 <time>으로 가림; - 전 / + 후):")
        for line in diff:
            print("   ", line)
    else:
        print("  출력 문장: 동일")

    changed = [c["cell"] for c in after if c["images"] and any(h not in before_images for h in c["images"])]
    if changed or len(before_images) != len(after_images):
        problems += 1
        print(f"  그림 바이트가 다른 셀(재작성 후 인덱스): {changed} · 시간 측정 그림이면 허용, 아니면 원인 설명 필요")
    before_others = [o for c in before for o in c["others"]]
    after_others = [o for c in after for o in c["others"]]
    if before_others != after_others:
        problems += 1
        print(f"  그 밖의 출력 차이: {before_others} → {after_others}")
    return 1 if problems else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    snap = sub.add_parser("snapshot", help="노트북에 저장된 출력을 JSON으로 저장")
    snap.add_argument("notebook", type=Path)
    snap.add_argument("--out", type=Path, default=None)
    comp = sub.add_parser("compare", help="스냅샷 JSON과 현재 노트북의 출력을 비교")
    comp.add_argument("before", type=Path)
    comp.add_argument("notebook", type=Path)
    args = parser.parse_args()

    if args.command == "snapshot":
        root = Path(__file__).resolve().parents[1]
        out = args.out or (root / "figures" / "check" / "snapshots" / f"{args.notebook.stem}.outputs.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        cells = snapshot(args.notebook)
        out.write_text(json.dumps(cells, ensure_ascii=False, indent=1), encoding="utf-8")
        images = sum(len(c["images"]) for c in cells)
        print(f"{args.notebook.name}: 코드 셀 {len(cells)} · 그림 {images} → {out}")
        return 0
    before = json.loads(args.before.read_text(encoding="utf-8"))
    return compare(before, snapshot(args.notebook), args.notebook.name)


if __name__ == "__main__":
    sys.exit(main())
