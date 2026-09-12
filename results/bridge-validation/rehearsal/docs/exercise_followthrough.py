from pathlib import Path
import json
import os
import tempfile
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

base=Path('/tmp/dlfs-bridges/rehearsal-docs')
workspace=base/'workspace'
python=str(base/'venv/bin/python')
os.environ.update(PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=str(base/'matplotlib'),
                 IPYTHONDIR=str(base/'ipython'))
probes={
 '00_math_to_numpy.ipynb': '''
def my_affine(X, W, b):
    N, D = X.shape
    K = W.shape[1]
    result = np.zeros((N, K))
    for n in range(N):
        for k in range(K):
            result[n, k] = b[k]
            for d in range(D):
                result[n, k] += X[n, d] * W[d, k]
    return result

check_three_outputs(my_affine)
rejected = False
try:
    check_three_outputs(lambda X, W, b: my_affine(X, W, b)[:, 0])
except AssertionError:
    rejected = True
assert rejected
print('DIRECT_EXERCISE_OK candidate=my_affine wrong_shape_rejected=True')
''',
 '01_gradients_and_learning.ipynb': '''
def my_mse_backward(residual):
    N, K = residual.shape
    result = np.zeros((N, K))
    for n in range(N):
        for k in range(K):
            result[n, k] = residual[n, k] / (N * K)
    return result

check_mse_backward(my_mse_backward)
rejected = False
try:
    check_mse_backward(lambda residual: residual / residual.shape[0])
except AssertionError:
    rejected = True
assert rejected
print('DIRECT_EXERCISE_OK candidate=my_mse_backward missing_K_rejected=True')
'''
}
records=[]
with tempfile.TemporaryDirectory(prefix='docs-exercise-kernel-') as temporary:
    kernel_root=Path(temporary)
    spec=kernel_root/'docs-venv'
    spec.mkdir()
    (spec/'kernel.json').write_text(json.dumps({'argv':[python,'-m','ipykernel_launcher','-f','{connection_file}'],
                                             'display_name':'Docs rehearsal fresh venv','language':'python'}))
    manager=KernelSpecManager(kernel_dirs=[str(kernel_root)])
    for filename,probe in probes.items():
        source=workspace/'notebooks/00_기초'/filename
        notebook=nbformat.read(source,as_version=4)
        for cell in notebook.cells:
            if cell.cell_type=='code':
                cell.outputs=[]
                cell.execution_count=None
        notebook.cells.append(nbformat.v4.new_code_cell(probe))
        km=KernelManager(kernel_name='docs-venv',kernel_spec_manager=manager,
                         transport='ipc',ip=str(kernel_root/filename))
        try:
            NotebookClient(notebook,km=km,timeout=180,
                           resources={'metadata':{'path':str(source.parent)}}).execute()
        finally:
            if km.has_kernel:km.shutdown_kernel(now=True)
        dest=base/f'exercise-{filename}'
        nbformat.write(notebook,dest)
        outputs=[o for c in notebook.cells for o in c.get('outputs',[])]
        streams=[o.text for o in outputs if o.output_type=='stream' and o.name=='stdout']
        errors=[dict(o) for o in outputs if o.output_type=='error']
        records.append({'source':str(source),'result':str(dest),'errors':errors,
                        'exercise_stdout':[s for s in streams if 'DIRECT_EXERCISE_OK' in s],
                        'code_cells_executed':sum(c.get('execution_count') is not None for c in notebook.cells)})
(base/'exercise_followthrough.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
print(json.dumps(records,ensure_ascii=False,indent=2))
