import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import time

os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['MPLBACKEND']='module://matplotlib_inline.backend_inline'
import nbformat
from nbclient import NotebookClient
from PIL import Image

root=Path('/home/junyeop/projects/deep-learning-from-scratch')
output=Path('/tmp/dlfs-architectures/rehearsal10')
source=root/'notebooks/01_아키텍처/10_gnn.ipynb'
source_bytes=source.read_bytes()
notebook=nbformat.reads(source_bytes.decode(),as_version=4)
original_cells=len(notebook.cells)
for cell in notebook.cells:
    if cell.cell_type=='code':
        cell.outputs=[]
        cell.execution_count=None
notebook.cells.append(nbformat.v4.new_markdown_cell('# 독립 실행자가 작성한 GNN 변형과 검산'))
notebook.cells.append(nbformat.v4.new_code_cell((output/'learner_exercises.py').read_text()))
start=time.monotonic()
client=NotebookClient(notebook,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(root)}})
client.execute()
elapsed=time.monotonic()-start
nbformat.write(notebook,output/'executed.ipynb')

figures=[]
streams=[]
errors=[]
for index,cell in enumerate(notebook.cells):
    if cell.cell_type!='code':continue
    for result in cell.outputs:
        if result.output_type=='stream':streams.append({'cell':index,'name':result.name,'text':result.text})
        if result.output_type=='error':errors.append({'cell':index,'data':dict(result)})
        if 'image/png' in result.get('data',{}):
            raw=result.data['image/png']
            if isinstance(raw,list):raw=''.join(raw)
            path=output/f'figure_{len(figures)+1:02d}_cell_{index}.png'
            path.write_bytes(base64.b64decode(raw))
            with Image.open(path) as picture:picture.verify()
            with Image.open(path) as picture:
                picture.load()
                figures.append({'cell':index,'path':str(path),'size':list(picture.size),
                                'decoded':True,'visually_reviewed':False})
record={'source':str(source),'source_sha256':hashlib.sha256(source_bytes).hexdigest(),
        'python_version':platform.python_version(),'fresh_kernel':True,
        'executed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'elapsed_seconds':elapsed,'original_cells':original_cells,
        'original_code_cells':sum(c.cell_type=='code' for c in notebook.cells[:original_cells]),
        'original_code_cells_executed':sum(c.cell_type=='code' and c.execution_count is not None for c in notebook.cells[:original_cells]),
        'source_unchanged_at_completion':source.read_bytes()==source_bytes,
        'errors':errors,'figure_count':len(figures),'figures':figures,'streams':streams,
        'glyph_warning_count':sum(s['text'].count('Glyph') for s in streams)}
(output/'execution.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
print(json.dumps({key:value for key,value in record.items() if key not in ['streams','figures']},ensure_ascii=False,indent=2))
print(json.dumps(figures,ensure_ascii=False,indent=2))
