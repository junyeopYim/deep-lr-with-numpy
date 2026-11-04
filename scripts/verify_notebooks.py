"""노트북을 현재 Python의 독립 커널에서 실행하고 출력·그림을 확인합니다.

사용: python scripts/verify_notebooks.py [노트북 경로 ...]
경로를 생략하면 foundation은 00_기초, architecture는 01_아키텍처,
bridge는 02_연결, composition은 03_구조조합, generative는 04_생성을 실행합니다.
"""

import argparse
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import nbformat
from notebook_validation import execute_notebook, execution_issues
from PIL import Image


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", type=Path)
    parser.add_argument("--suite", default="foundation", choices=("foundation", "architecture", "bridge", "composition", "generative"),
                        help="실행 기록과 그림을 저장할 학습 단위")
    args = parser.parse_args()
    folder = {"foundation": "00_기초", "architecture": "01_아키텍처", "bridge": "02_연결",
              "composition": "03_구조조합", "generative": "04_생성"}[args.suite]
    paths = args.notebooks or sorted((root / "notebooks" / folder).rglob("*.ipynb"))
    paths = [p.resolve() for p in paths if ".ipynb_checkpoints" not in p.parts]
    if not paths:
        parser.error("실행할 노트북 경로를 지정해 주세요.")
    report_dir = root / "results" / f"{args.suite}-validation"
    figure_dir = root / "figures" / f"{args.suite}-validation"
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    report_path = report_dir / "execution.json"
    report_path.write_text("[]")  # 이전 실행의 성공 기록을 이번 실행으로 오인하지 않게 합니다.
    for path in paths:
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        start = time.perf_counter()
        print(f"실행: {path.relative_to(root)}", flush=True)
        try:
            execute_notebook(nb, path)
            issues = execution_issues(nb)
            if any(issues.values()):
                raise RuntimeError(f"실행 결과를 확인해 주세요: {path.name}\n{issues}")
            cells = [c for c in nb.cells if c.cell_type == "code" and c.source.strip()]
            outputs = [o for cell in cells for o in cell.get("outputs", [])]
            figures = []
            for output in outputs:
                raw = output.get("data", {}).get("image/png")
                if raw:
                    dest = figure_dir / f"{path.stem}-{len(figures) + 1:02}.png"
                    dest.write_bytes(base64.b64decode(raw))
                    with Image.open(dest) as img:
                        img.verify()
                    figures.append(str(dest.relative_to(root)))
        except Exception:
            nb.metadata.setdefault("learning", {})["status"] = "failed"
            nbformat.write(nb, report_dir / f"{path.stem}-failed.ipynb")
            raise
        nb.metadata.setdefault("learning", {})["status"] = "executed"
        nbformat.write(nb, path)
        report = {
            "notebook": str(path.relative_to(root)),
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0],
            "code_cells": len(cells), "figures": figures,
            "seconds": round(time.perf_counter() - start, 3),
            "errors": 0, "warnings": [],
            "stdout": "\n".join(o.text for o in outputs if o.output_type == "stream" and o.name == "stdout"),
        }
        reports.append(report)
        report_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2))
        print(f"확인: 코드 {len(cells)}개 · 그림 {len(figures)}개 · {report['seconds']}초", flush=True)


if __name__ == "__main__":
    main()
