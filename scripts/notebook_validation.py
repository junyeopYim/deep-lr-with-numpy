"""노트북 검증 명령들이 공유하는 새 커널 실행과 실패 판정."""

import json
import os
from pathlib import Path
import sys
import tempfile

from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


def execute_notebook(nb, path: Path, timeout: int = 180):
    """이 스크립트와 같은 Python의 새 커널에서 실행합니다. 이전 출력은 지웁니다."""
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
            cell.metadata.pop("execution", None)
    nb.metadata.setdefault("learning", {})["status"] = "running"
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with tempfile.TemporaryDirectory(prefix="dlfs-kernel-") as temporary:
        kernel_root = Path(temporary)
        spec_dir = kernel_root / "dlfs-current"
        spec_dir.mkdir()
        (spec_dir / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Current Python", "language": "python",
        }))
        specs = KernelSpecManager(kernel_dirs=[str(kernel_root)])
        connection = {"transport": "ipc", "ip": str(kernel_root / "connection")} if os.name == "posix" else {}
        # 비동기 채널이어야 응답을 기다리는 동안에도 커널 생존 여부를 확인할 수 있습니다.
        manager = KernelManager(kernel_name="dlfs-current", kernel_spec_manager=specs,
                                client_class="jupyter_client.asynchronous.client.AsyncKernelClient", **connection)
        try:
            NotebookClient(nb, km=manager, timeout=timeout,
                           resources={"metadata": {"path": str(path.parent)}}).execute(cleanup_kc=True)
        except Exception:
            nb.metadata["learning"]["status"] = "failed"
            raise
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)


def execution_issues(nb, *, require_answers=False):
    """오류·stderr·미실행 셀·연습 실패를 모읍니다. 미작성 연습은 기본으로 허용합니다."""
    issues = {"errors": [], "stderr": [], "unexecuted": [], "exercise_failures": []}
    for index, cell in enumerate(nb.cells):
        if cell.cell_type != "code" or not cell.source.strip():
            continue
        if cell.execution_count is None:
            issues["unexecuted"].append(index)
        stdout = []
        for output in cell.get("outputs", []):
            if output.output_type == "error":
                issues["errors"].append(f"셀 {index}: {output.ename}: {output.evalue}")
            elif output.output_type == "stream":
                if output.name == "stderr":
                    issues["stderr"].append(f"셀 {index}: {output.text}")
                else:
                    stdout.append(output.text)
        for line in "".join(stdout).splitlines():
            if "✗" in line or (require_answers and "☐" in line):
                issues["exercise_failures"].append(f"셀 {index}: {line}")
    return issues
