"""연습 뼈대의 `raise NotImplementedError`를 '정답 보기'의 줄로 바꿔 실행하고 모든 검사가 통과하는지 봅니다.

사용: python scripts/verify_exercises.py notebooks/00_기초/00b_vectorization.ipynb [...]

노트북 파일은 바꾸지 않습니다(메모리에서만 채워 실행). 정답은 각 연습 셀 뒤의
`<details>` 마크다운 안 ```python 블록에서 읽습니다.
"""

import os
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def verify(path: Path) -> int:
    nb = nbformat.read(path, as_version=4)
    cells = nb.cells
    filled = 0
    for i, cell in enumerate(cells):
        if cell.cell_type != "code" or "raise NotImplementedError" not in cell.source:
            continue
        j = i + 1
        while j < len(cells) and not (cells[j].cell_type == "markdown" and "<details>" in cells[j].source):
            j += 1
        if j == len(cells):
            print(f"{path.name}: 셀 {i}의 정답 블록을 찾지 못했습니다")
            return 1
        match = re.search(r"```python\n(.*?)```", cells[j].source, re.S)
        answer = match.group(1).strip("\n").split("\n")
        lines = cell.source.split("\n")
        k = next(idx for idx, line in enumerate(lines) if "raise NotImplementedError" in line)
        indent = lines[k][: len(lines[k]) - len(lines[k].lstrip())]
        lines[k:k + 1] = [indent + line for line in answer]
        cell.source = "\n".join(lines)
        filled += 1
    NotebookClient(nb, timeout=600, kernel_name="python3",
                   resources={"metadata": {"path": str(path.parent)}}).execute()
    failures = []
    for i, cell in enumerate(cells):
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            text = "".join(output.get("text", "")) if output.get("output_type") == "stream" else ""
            if "☐" in text or "✗" in text:
                failures.append((i, text.strip()[:120]))
            if output.get("output_type") == "error":
                failures.append((i, output.get("ename")))
    print(f"{path.name}: 채운 연습 {filled}개 · 실패 {len(failures)}개 {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(max(verify(Path(p).resolve()) for p in sys.argv[1:]) if len(sys.argv) > 1 else 2)
