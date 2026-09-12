from pathlib import Path
import base64, hashlib, json, os, sys, time
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ.pop('MPLBACKEND', None)
import nbformat
from nbclient import NotebookClient
from PIL import Image
ROOT=Path('/home/junyeop/projects/deep-learning-from-scratch')
OUT=Path('/tmp/dlfs-architectures/rehearsal05-07')
name=sys.argv[1]
source=ROOT/'notebooks/01_아키텍처'/(name+'.ipynb')
raw=source.read_bytes()
source_hash=hashlib.sha256(raw).hexdigest()
copy=OUT/(name+'.input.ipynb')
copy.write_bytes(raw)
nb=nbformat.read(copy,as_version=4)
original_count=len(nb.cells)
for cell in nb.cells:
    if cell.cell_type=='code':
        cell.outputs=[]
        cell.execution_count=None
nb.cells.append(nbformat.v4.new_code_cell((OUT/(name+'_exercise.py')).read_text()))
cell_starts={}
cell_times=[]
def started(cell,cell_index,**kwargs):
    cell_starts[cell_index]=time.perf_counter()
    print(f'{name} cell {cell_index} start',flush=True)
def executed(cell,cell_index,**kwargs):
    elapsed=time.perf_counter()-cell_starts[cell_index]
    cell_times.append({'cell':cell_index,'seconds':elapsed})
    print(f'{name} cell {cell_index} done {elapsed:.3f}s',flush=True)
client=NotebookClient(nb,timeout=300,kernel_name='python3',resources={'metadata':{'path':str(source.parent)}},on_cell_execute=started,on_cell_executed=executed)
start=time.perf_counter()
try:
    client.execute()
except Exception:
    nbformat.write(nb,OUT/(name+'.failed.ipynb'))
    raise
seconds=time.perf_counter()-start
nbformat.write(nb,OUT/(name+'.executed.ipynb'))
streams=[]
figures=[]
for ci,cell in enumerate(nb.cells):
    for oi,output in enumerate(cell.get('outputs',[])):
        if output.output_type=='stream':
            streams.append({'cell':ci,'name':output.name,'text':output.text})
        if 'image/png' in output.get('data',{}):
            path=OUT/f'{name}_cell{ci:02d}_{oi}.png'
            path.write_bytes(base64.b64decode(output.data['image/png']))
            with Image.open(path) as im:
                im.load()
                width,height=im.size
            figures.append({'cell':ci,'path':str(path),'width':width,'height':height,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
metrics={}
for entry in streams:
    for line in entry['text'].splitlines():
        if line.startswith('REHEARSAL_METRICS='):
            metrics=json.loads(line.split('=',1)[1])
result={'notebook':str(source.relative_to(ROOT)),'source_sha256':source_hash,'source_unchanged':hashlib.sha256(source.read_bytes()).hexdigest()==source_hash,'fresh_kernel':True,'original_cell_count':original_count,'original_code_cells_executed':sum(c.cell_type=='code' and c.execution_count is not None for c in nb.cells[:original_count]),'appended_exercise_cells':1,'elapsed_seconds':seconds,'cell_times':cell_times,'figures':figures,'stderr':[s for s in streams if s['name']=='stderr'],'metrics':metrics}
(OUT/(name+'_metrics.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
(OUT/(name+'_execution.log')).write_text('\n'.join(f"CELL {s['cell']} [{s['name']}]\n{s['text']}" for s in streams))
print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
