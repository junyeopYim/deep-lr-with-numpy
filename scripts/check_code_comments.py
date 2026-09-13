"""노트북 코드 셀이 docs/WRITING.md 7절(코드 주석 규칙)을 지키는지 실행 없이 점검합니다.

사용: python scripts/check_code_comments.py notebooks/00_기초/00_math_to_numpy.ipynb [...]

보고하는 것:
- docstring 없는 def (연습 뼈대·검사 함수 포함)
- 코드 줄(주석·docstring·빈 줄 제외)이 25줄을 넘는 셀 (class 하나 또는 def 하나만 있는 정의 셀은 나눌 수 없으므로 제외)
- 한 줄에 세미콜론으로 이어 붙인 두 문장
- 메시지 없는 assert, 줄 끝 `# 검산:` 주석이 없는 np.testing.assert_*
- lambda
- 세 셀 이상 위에서 만든 이름을 `# 쓰는 것:` 표시 없이 쓰는 셀 (import 한 이름은 제외)

위반이 하나라도 있으면 종료 코드 1입니다. 노트북 파일은 바꾸지 않습니다.
"""

import argparse
import ast
import builtins
import io
import re
import sys
import tokenize
from pathlib import Path

import nbformat

MAX_CODE_LINES = 25
FAR_CELLS = 3
USES_MARK = "쓰는 것"


def code_line_count(src, tree):
    """주석·빈 줄·docstring을 뺀 줄 수."""
    doc_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                doc_lines.update(range(body[0].lineno, body[0].end_lineno + 1))
    count = 0
    for number, line in enumerate(src.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or number in doc_lines:
            continue
        count += 1
    return count


def tokens_of(src):
    try:
        return list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, SyntaxError):
        return []


def comment_lines(tokens):
    """{줄 번호: 주석 문자열}"""
    return {tok.start[0]: tok.string for tok in tokens if tok.type == tokenize.COMMENT}


def semicolon_lines(tokens):
    return sorted({tok.start[0] for tok in tokens if tok.type == tokenize.OP and tok.string == ";"})


def names_defined(tree):
    """이 셀이 만드는 이름(대입·함수·클래스·import·for 변수·with)."""
    defined, imported = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            defined.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported.add(alias.asname or alias.name)
    return defined, imported


def names_loaded_globally(tree):
    """함수 매개변수·지역 변수를 뺀, 바깥에서 와야 하는 이름.
    연습 뼈대(exercise_*) 안은 읽는 사람이 채우는 자리라 보지 않습니다."""
    loaded = set()

    def visit(node, locals_):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("exercise_"):
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            args = node.args
            params = {a.arg for a in args.args + args.kwonlyargs + args.posonlyargs}
            if args.vararg:
                params.add(args.vararg.arg)
            if args.kwarg:
                params.add(args.kwarg.arg)
            stores = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
            locals_ = locals_ | params | stores
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in locals_:
            loaded.add(node.id)
        for child in ast.iter_child_nodes(node):
            visit(child, locals_)

    visit(tree, frozenset())
    return loaded


def is_plot_only(tree):
    """그림 함수(plot_*, draw_*) 호출만 있는 셀."""
    return bool(tree.body) and all(
        isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
        and getattr(stmt.value.func, "id", "").startswith(("plot_", "draw_"))
        for stmt in tree.body
    )


def check(path: Path) -> list[str]:
    nb = nbformat.read(path, as_version=4)
    problems = []
    defined_at = {}          # 이름 → 마지막으로 만든 코드 셀 순번
    imported_names = set(dir(builtins))
    order = 0
    for index, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        src = cell.source
        try:
            tree = ast.parse(src)
        except SyntaxError as error:
            problems.append(f"셀 {index}: 구문 오류 {error}")
            continue
        tokens = tokens_of(src)
        comments = comment_lines(tokens)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and ast.get_docstring(node) is None:
                problems.append(f"셀 {index}: def {node.name} 에 docstring 없음")
            if isinstance(node, ast.Lambda):
                problems.append(f"셀 {index}: lambda (줄 {node.lineno}) → 이름 있는 def 로")
            if isinstance(node, ast.Assert) and node.msg is None:
                problems.append(f"셀 {index}: 메시지 없는 assert (줄 {node.lineno})")
            if isinstance(node, ast.Call):
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if name.startswith("assert_") and "검산" not in comments.get(node.lineno, ""):
                    problems.append(f"셀 {index}: {name} 에 `# 검산:` 주석 없음 (줄 {node.lineno})")

        lines = code_line_count(src, tree)
        single_definition = len(tree.body) == 1 and isinstance(tree.body[0], (ast.ClassDef, ast.FunctionDef))
        if lines > MAX_CODE_LINES and not single_definition:   # 정의 하나뿐인 셀은 셀로 나눌 수 없으므로 단계 번호 주석으로 대신합니다
            problems.append(f"셀 {index}: 코드 {lines}줄 > {MAX_CODE_LINES}줄 → 정의/실행/검산으로 나누기")
        for line in semicolon_lines(tokens):
            problems.append(f"셀 {index}: 한 줄에 두 문장 ';' (줄 {line})")

        defined, imported = names_defined(tree)
        imported_names |= imported
        uses_comment = " ".join(text for text in comments.values() if USES_MARK in text)
        far = []
        for name in sorted(names_loaded_globally(tree) if not is_plot_only(tree) else ()):
            if name in imported_names or name in defined:
                continue
            birth = defined_at.get(name)
            if birth is not None and order - birth > FAR_CELLS and not re.search(rf"\b{re.escape(name)}\b", uses_comment):
                far.append(name)
        if far:
            problems.append(f"셀 {index}: 멀리서 온 이름에 `# {USES_MARK}:` 표시 없음 → {', '.join(far)}")
        for name in defined:
            defined_at[name] = order
        order += 1
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("notebooks", nargs="+", type=Path)
    args = parser.parse_args()
    status = 0
    for path in args.notebooks:
        problems = check(path)
        print(f"{path.name}: {'통과' if not problems else f'{len(problems)}건'}")
        for problem in problems:
            print("  -", problem)
        status |= bool(problems)
    return status


if __name__ == "__main__":
    sys.exit(main())
