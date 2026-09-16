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


def insert_before(nb, key, parts, cell_type=None):
    """key 가 들어 있는 셀 바로 앞에 parts = [("code"|"markdown", source), ...] 셀들을 끼웁니다.
    절 끝에 연습을 둘 때 key 로 다음 절의 제목("## 🔑 3. ")을 주면 그 앞, 즉 앞 절의 끝에 들어갑니다."""
    i = find(nb, key, cell_type)
    new_cells = []
    for kind, src in parts:
        src = src.rstrip("\n")
        new_cells.append(new_code_cell(src) if kind == "code" else new_markdown_cell(src))
    nb.cells[i:i] = new_cells
    return i


HOW_TO_SOLVE = (
    "연습은 코딩 테스트처럼 **함수 뼈대가 미리 있고 핵심 한두 줄만 비어 있습니다.** "
    "`raise NotImplementedError` 줄을 지우고 그 자리에 코드를 쓴 뒤, 셀을 실행하고 바로 아래 검사 셀을 실행하세요. "
    "아직 채우지 않았으면 안내문만 나오고, 값이 다르면 \"손계산과 다릅니다\"가 나옵니다. 막히면 그 아래 \"정답 보기\"를 펼칩니다."
)


def exercise_cells(label, title, instruction, skeleton, check, answer, intro=False):
    """절 연습 네 셀을 WRITING.md 5절 형식으로 만듭니다: ✏️ 제목·안내(markdown) / 뼈대(code) / 검사와 run_check(code) / 접힌 정답(markdown).
    label: "3" 또는 "3-2" 처럼 절 번호, title: 제목 한 줄, instruction: 무엇을 채우는지 한두 문장,
    skeleton: exercise_* 함수 소스(raise NotImplementedError 한 줄 포함), check: check_* 함수와 run_check(...) 줄,
    answer: raise 줄 자리에 들어갈 정답 코드(들여쓰기 없이). intro=True 면 노트북의 첫 연습이라 푸는 법 문단을 붙입니다."""
    head = f"### ✏️ 연습 {label} · {title}\n\n"
    if intro:
        head += HOW_TO_SOLVE + "\n\n"
    head += instruction.strip("\n")
    solution = (
        "📎 **정답** · 막히면 펼쳐 보고, 다시 접고 스스로 써 보세요.\n\n"
        "<details>\n<summary>정답 보기</summary>\n\n"
        f"```python\n{answer.strip(chr(10))}\n```\n\n</details>"
    )
    return [("markdown", head), ("code", skeleton), ("code", check), ("markdown", solution)]
