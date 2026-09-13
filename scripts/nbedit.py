r"""노트북 셀을 고유 문자열로 찾아 바꾸거나 여러 셀로 나누는 도우미. 노트북 JSON을 손으로 고치지 않기 위한 것입니다.

scratchpad의 재작성 스크립트에서 이렇게 씁니다::

    import sys; sys.path.insert(0, "scripts")
    from nbedit import load, save, replace, edit, append, split

    nb = load("notebooks/00_기초/01_gradients_and_learning.ipynb")
    replace(nb, "def mse_forward(", NEW_SOURCE)                    # 셀 전체 교체. NEW_SOURCE는 새 셀 소스 문자열
    edit(nb, "def check_slope(", "    assert a == b\n", '    assert a == b, "…이어야 합니다"\n')   # 셀 안 부분 교체
    append(nb, "## 🔑 2. ", "> **코드** · `_`: …")                 # 마크다운 셀 끝에 빈 줄 하나 두고 덧붙임
    split(nb, "def train(", [("code", DEF_SOURCE), ("code", RUN_SOURCE)])   # 한 셀을 정의 셀·실행 셀로
    save(nb, "notebooks/00_기초/01_gradients_and_learning.ipynb")

key 는 그 노트북에서 정확히 한 셀에만 들어 있는 문자열이어야 합니다(아니면 AssertionError).
"""

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell


def load(path):
    return nbformat.read(str(path), as_version=4)


def save(nb, path):
    nbformat.write(nb, str(path))


def find(nb, key, cell_type=None):
    hits = [i for i, c in enumerate(nb.cells) if key in c.source and (cell_type is None or c.cell_type == cell_type)]
    assert len(hits) == 1, f"{key!r}: {len(hits)}개 셀과 일치 {hits}"
    return hits[0]


def replace(nb, key, new_source, cell_type=None):
    """key 가 들어 있는 셀의 소스 전체를 new_source 로 바꿉니다."""
    i = find(nb, key, cell_type)
    nb.cells[i].source = new_source.rstrip("\n")
    return i


def edit(nb, key, old, new, cell_type=None):
    """key 가 들어 있는 셀 안에서 old(정확히 한 번 있어야 함)를 new 로 바꿉니다."""
    i = find(nb, key, cell_type)
    src = nb.cells[i].source
    assert src.count(old) == 1, f"{old!r}: 셀 {i}에 {src.count(old)}번 등장"
    nb.cells[i].source = src.replace(old, new)
    return i


def append(nb, key, extra, cell_type="markdown"):
    """key 가 들어 있는 셀 끝에 빈 줄 하나를 두고 extra 를 덧붙입니다."""
    i = find(nb, key, cell_type)
    nb.cells[i].source = nb.cells[i].source.rstrip("\n") + "\n\n" + extra.strip("\n")
    return i


def split(nb, key, parts, cell_type="code"):
    """key 가 들어 있는 셀 하나를 parts = [("code"|"markdown", source), ...] 여러 셀로 바꿉니다."""
    i = find(nb, key, cell_type)
    new_cells = []
    for kind, src in parts:
        src = src.rstrip("\n")
        new_cells.append(new_code_cell(src) if kind == "code" else new_markdown_cell(src))
    nb.cells[i:i + 1] = new_cells
    return i
