"""검증 명령이 실제 커널 종료와 출력으로 보고된 연습 실패도 거부하는지 검사합니다."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

import nbformat
import pytest

from scripts.notebook_validation import execution_issues

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def workspace(tmp_path):
    """검증 명령의 쓰기 대상이 저장소에 닿지 않는 작은 작업 폴더."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("check_notebook.py", "verify_notebooks.py", "verify_exercises.py", "notebook_validation.py"):
        shutil.copy(ROOT / "scripts" / name, scripts / name)
    return tmp_path


def make_notebook(workspace, cells):
    """오래된 성공 출력도 지워지는지 확인하기 위해 실행된 모양의 파일을 만듭니다."""
    nb = nbformat.v4.new_notebook(cells=cells, metadata={"learning": {"status": "executed"}})
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.execution_count = 1
            cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text="OLD SUCCESS\n")]
    path = workspace / "probe.ipynb"
    nbformat.write(nb, path)
    return path


def run_script(workspace, name, notebook):
    return subprocess.run([sys.executable, str(workspace / "scripts" / name), str(notebook)],
                          cwd=workspace, text=True, capture_output=True, timeout=45)


def exercise_cells(answer):
    """실제 run_check가 검사 예외를 출력으로 바꾸는 최소 연습."""
    return [nbformat.v4.new_code_cell(f"import sys\nsys.path.insert(0, {str(ROOT)!r})\nfrom src.utils import run_check"),
            nbformat.v4.new_code_cell(f"def exercise_answer():\n    return {answer}"),
            nbformat.v4.new_code_cell("def check_answer(candidate):\n    assert candidate() == 1\n    print('통과')\nrun_check(check_answer, exercise_answer)")]


@pytest.mark.parametrize("command", ["check_notebook.py", "verify_notebooks.py"])
def test_cli_rejects_wrong_exercise(workspace, command):
    path = make_notebook(workspace, exercise_cells(2))
    result = run_script(workspace, command, path)
    assert result.returncode != 0, result.stdout + result.stderr
    if command == "check_notebook.py":
        assert nbformat.read(path, 4).metadata.learning.status == "failed"
    else:
        report = workspace / "results/foundation-validation/execution.json"
        assert json.loads(report.read_text()) == []
        failed = nbformat.read(report.parent / "probe-failed.ipynb", 4)
        assert failed.metadata.learning.status == "failed"


def test_check_notebook_rejects_dead_kernel_and_discards_stale_output(workspace):
    path = make_notebook(workspace, [nbformat.v4.new_code_cell("import os\nos._exit(3)"),
                                     nbformat.v4.new_code_cell("print('FINISHED')")])
    result = run_script(workspace, "check_notebook.py", path)
    assert result.returncode != 0, result.stdout + result.stderr
    nb = nbformat.read(path, 4)
    assert nb.metadata.learning.status == "failed"
    assert nb.cells[1].execution_count is None
    assert not nb.cells[1].outputs


def test_new_notebook_preserves_original_execution_error(workspace):
    path = workspace / "new.ipynb"
    nbformat.write(nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_code_cell("raise ValueError('deliberate failure')")]), path)
    result = run_script(workspace, "check_notebook.py", path)
    assert result.returncode != 0
    assert "deliberate failure" in result.stdout
    assert "has no attribute 'status'" not in result.stdout
    assert nbformat.read(path, 4).metadata.learning.status == "failed"


@pytest.mark.parametrize("command", ["check_notebook.py", "verify_notebooks.py"])
def test_original_allows_unfilled_exercise_in_current_python(workspace, command):
    cells = exercise_cells(1)
    cells[1].source = "def exercise_answer():\n    raise NotImplementedError"
    cells.append(nbformat.v4.new_code_cell(f"assert sys.executable == {sys.executable!r}"))
    path = make_notebook(workspace, cells)
    result = run_script(workspace, command, path)
    assert result.returncode == 0, result.stdout + result.stderr
    nb = nbformat.read(path, 4)
    assert nb.metadata.learning.status == "executed"
    assert all(c.execution_count is not None for c in nb.cells)


@pytest.mark.parametrize("answer,expected", [(1, 0), (2, 1)])
def test_exercise_verifier_checks_answers_without_writing_source(workspace, answer, expected):
    cells = exercise_cells(1)
    cells[1].source = "def exercise_answer():\n    raise NotImplementedError"
    cells.append(nbformat.v4.new_markdown_cell(f"<details>\n```python\nreturn {answer}\n```\n</details>"))
    path = make_notebook(workspace, cells)
    before = path.read_bytes()
    result = run_script(workspace, "verify_exercises.py", path)
    assert result.returncode == expected, result.stdout + result.stderr
    assert path.read_bytes() == before


def test_execution_issues_distinguishes_placeholders_warnings_and_unexecuted_cells():
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("run_check(...)"),
                                       nbformat.v4.new_code_cell(""),
                                       nbformat.v4.new_code_cell("print('later')")])
    nb.cells[0].execution_count = 1
    nb.cells[0].outputs = [nbformat.v4.new_output("stream", name="stdout", text="☐ 미작성\n"),
                           nbformat.v4.new_output("stream", name="stderr", text="Glyph missing\n")]
    issues = execution_issues(nb)
    assert issues["unexecuted"] == [2]
    assert issues["stderr"]
    assert not issues["exercise_failures"]
    assert execution_issues(nb, require_answers=True)["exercise_failures"]


@pytest.mark.parametrize("suite,folder", [
    ("representation", "05_표현학습"), ("language", "06_언어모델"),
    ("reinforcement", "07_강화학습"), ("capstone", "08_종합"), ("all", "08_종합"),
])
def test_new_suites_discover_notebooks_without_checkpoints(workspace, suite, folder):
    """새 단원도 경로 없이 실행되며 checkpoint 복사본·다른 suite는 섞이지 않아야 합니다."""
    path = workspace / "notebooks" / folder / "lesson.ipynb"
    path.parent.mkdir(parents=True)
    nbformat.write(nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_code_cell("print('NEW SUITE OK')")]), path)
    checkpoint = path.parent / ".ipynb_checkpoints" / "lesson-checkpoint.ipynb"
    checkpoint.parent.mkdir()
    nbformat.write(nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_code_cell("raise RuntimeError('checkpoint must not execute')")]), checkpoint)
    other = workspace / "notebooks" / "00_기초" / "other.ipynb"
    other.parent.mkdir()
    nbformat.write(nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_code_cell("print('FOUNDATION OK')")]), other)
    result = subprocess.run([sys.executable, str(workspace / "scripts/verify_notebooks.py"),
                             "--suite", suite], cwd=workspace, text=True, capture_output=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    reports = json.loads((workspace / "results" / f"{suite}-validation/execution.json").read_text())
    expected = {str(path.relative_to(workspace))}
    if suite == "all":
        expected.add(str(other.relative_to(workspace)))
    assert {item["notebook"] for item in reports} == expected
    assert all(item["errors"] == 0 and not item["warnings"] for item in reports)
    assert "NEW SUITE OK" in next(item["stdout"] for item in reports if item["notebook"].endswith("lesson.ipynb"))
