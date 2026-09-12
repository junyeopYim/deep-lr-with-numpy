import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['MPLCONFIGDIR'] = '/tmp/dlfs-architectures/rehearsal05/matplotlib'
os.environ['IPYTHONDIR'] = '/tmp/dlfs-architectures/rehearsal05/ipython'
import sys, json, pathlib, hashlib, time, base64, io
sys.dont_write_bytecode = True
import nbformat
from nbclient import NotebookClient
from PIL import Image
root = pathlib.Path('/home/junyeop/projects/deep-learning-from-scratch')
out = pathlib.Path('/tmp/dlfs-architectures/rehearsal05')
source = root/'notebooks/01_아키텍처/05_cnn.ipynb'
raw = source.read_bytes()
(out/'source.ipynb').write_bytes(raw)
notebook = nbformat.reads(raw.decode(), as_version=4)
original_cells = len(notebook.cells)
original_code_cells = sum(c.cell_type == 'code' for c in notebook.cells)
extra = '''
# Reconstruct each derivative directly from the scalar convolution in section 1.
# Cached P holds the original input patches; no reference backward is called.
def my_conv_backward(dY, cache):
    shape, K, P, stride, padding = cache
    N, C, H, W = shape
    F, _, kh, kw = K.shape
    oh, ow = dY.shape[2:]
    patches = P.reshape(N, oh, ow, C, kh, kw)
    dXp = np.zeros((N, C, H + 2 * padding, W + 2 * padding))
    dK = np.zeros_like(K)
    db = np.zeros(F)
    for n in range(N):
        for f in range(F):
            for i in range(oh):
                for j in range(ow):
                    g = dY[n, f, i, j]
                    db[f] += g
                    dK[f] += g * patches[n, i, j]
                    row, col = i * stride, j * stride
                    dXp[n, :, row:row + kh, col:col + kw] += g * K[f]
    return dXp[:, :, padding:padding + H, padding:padding + W], dK, db

candidate_calls = {'conv': 0, 'pool': 0, 'wrong_overlap': 0, 'empty': 0, 'partial': 0}
def counted_my_conv(dY, cache):
    candidate_calls['conv'] += 1
    return my_conv_backward(dY, cache)
learner_conv_errors = check_conv_backward(counted_my_conv)
assert candidate_calls['conv'] == 2 and len(learner_conv_errors) == 6

# A single intentional defect: overwrite a spatial patch instead of accumulating.
def wrong_overwrite(dY, cache):
    candidate_calls['wrong_overlap'] += 1
    shape, K, P, stride, padding = cache
    N, C, H, W = shape
    F, _, kh, kw = K.shape
    oh, ow = dY.shape[2:]
    _, correct_dK, correct_db = my_conv_backward(dY, cache)
    dXp = np.zeros((N, C, H + 2 * padding, W + 2 * padding))
    for n in range(N):
        for i in range(oh):
            for j in range(ow):
                patch_gradient = np.zeros((C, kh, kw))
                for f in range(F):
                    patch_gradient += dY[n, f, i, j] * K[f]
                row, col = i * stride, j * stride
                dXp[n, :, row:row + kh, col:col + kw] = patch_gradient
    return dXp[:, :, padding:padding + H, padding:padding + W], correct_dK, correct_db

wrong_overlap_rejected = False
try:
    check_conv_backward(wrong_overwrite)
except AssertionError as exc:
    wrong_overlap_rejected = True
    wrong_overlap_error = repr(exc)
assert wrong_overlap_rejected

# Independently distribute each mean pooling output derivative to its four pixels.
def my_mean_pool_backward(dY, cache):
    candidate_calls['pool'] += 1
    shape, mode, indices = cache
    N, C, H, W = shape
    assert mode == 'mean'
    dX = np.zeros(shape)
    for n in range(N):
        for c in range(C):
            for i in range(H // 2):
                for j in range(W // 2):
                    for u in range(2):
                        for v in range(2):
                            dX[n, c, 2*i + u, 2*j + v] = dY[n, c, i, j] / 4
    return dX
learner_pool_result = check_mean_pool_backward(my_mean_pool_backward)
assert candidate_calls['pool'] == 1

wrong_pool_rejected = False
try:
    check_mean_pool_backward(lambda dY, cache: 4 * my_mean_pool_backward(dY, cache))
except AssertionError as exc:
    wrong_pool_rejected = True
    wrong_pool_error = repr(exc)
assert wrong_pool_rejected

# The contract requires three derivatives. Probe whether the checker enforces it.
def empty_candidate(dY, cache):
    candidate_calls['empty'] += 1
    return ()
def partial_candidate(dY, cache):
    candidate_calls['partial'] += 1
    return (my_conv_backward(dY, cache)[0],)
contract_probes = {}
for name, function in [('empty', empty_candidate), ('only_dX', partial_candidate)]:
    try:
        returned = check_conv_backward(function)
        contract_probes[name] = {'rejected': False, 'returned_errors': returned}
    except (AssertionError, ValueError, TypeError) as exc:
        contract_probes[name] = {'rejected': True, 'exception': repr(exc)}

# Unequal/signed upstream pooling values and first-maximum tie convention.
pool_audit_rng = np.random.default_rng(581)
pool_audit_X = pool_audit_rng.normal(size=(2, 3, 4, 6))
pool_gradient_errors = {}
for pool_mode in ['mean', 'max']:
    pool_output, pool_cache_audit = pool_forward(pool_audit_X, pool_mode)
    pool_G = pool_audit_rng.normal(size=pool_output.shape)
    pool_numeric = numerical_gradient(lambda: float(np.sum(pool_forward(pool_audit_X, pool_mode)[0] * pool_G)), pool_audit_X)
    pool_gradient_errors[pool_mode] = float(rel_error(pool_backward(pool_G, pool_cache_audit), pool_numeric))
    assert pool_gradient_errors[pool_mode] < 2e-6
max_tie_X = np.array([[[[2., 2.], [1., 0.]]]])
max_tie_Y, max_tie_cache = pool_forward(max_tie_X, 'max')
max_tie_gradient = pool_backward(np.ones_like(max_tie_Y), max_tie_cache)
np.testing.assert_array_equal(max_tie_gradient, [[[[1., 0.], [0., 0.]]]])

# Change the first hand example's filter, and compare the output to hand arithmetic.
changed_K = np.array([[[[1., 2.], [-1., 0.]]]])
changed_output = conv_forward(small_X, changed_K, np.zeros(1))[0]
np.testing.assert_allclose(changed_output[0,0], [[1., 3.], [7., 9.]])

# Record baseline before the variation, preserving its trained parameters.
baseline_parameters = sum(value.size for value in p.values())
baseline_train_logits, _ = cnn_forward(X_train, p)
baseline = {
    'filters': 6, 'head_input_dimension': 54, 'parameter_count': baseline_parameters,
    'steps': 600, 'batch_size': 50, 'train_samples': len(y_train), 'valid_samples': len(y_valid),
    'train_initial_ce': float(history['train'][0]), 'train_final_ce': float(history['train'][-1]),
    'valid_initial_ce': float(history['valid'][0]), 'valid_final_ce': float(history['valid'][-1]),
    'train_accuracy': float(np.mean(baseline_train_logits.argmax(axis=1) == y_train)),
    'valid_accuracy': float(valid_accuracy),
    'parameter_shapes': {key: list(value.shape) for key, value in p.items()},
    'history': {key: list(map(float, values)) for key, values in history.items()}}

# Section 7 variation: F=3 -> flattened features 3*3*3=27 and 310 parameters.
# Use the same data, parameter RNG seed, batch RNG seed, 600 updates, and Adam rate.
import time
variation_started = time.time()
variation_rng = np.random.default_rng(503)
filters = 3
head_dimension = filters * 3 * 3
p3 = {'K': variation_rng.normal(size=(filters,1,3,3)) * np.sqrt(2/9),
      'c': np.zeros(filters),
      'W': variation_rng.normal(size=(head_dimension,10)) * np.sqrt(1/head_dimension),
      'b': np.zeros(10)}
assert head_dimension == 27 and sum(value.size for value in p3.values()) == 310
shape_logits3, shape_cache3 = cnn_forward(X_train[:2], p3)
assert shape_logits3.shape == (2,10)
assert shape_cache3[-1] == (2,3,3,3)
assert shape_cache3[-2].shape == (2,27)
state3 = {key: (np.zeros_like(value), np.zeros_like(value)) for key, value in p3.items()}
variation_batch_rng = np.random.default_rng(505)
history3 = {'step': [], 'train': [], 'valid': []}
for variation_step in range(601):
    if variation_step % 50 == 0:
        train_logits3, _ = cnn_forward(X_train, p3)
        valid_logits3, _ = cnn_forward(X_valid, p3)
        history3['step'].append(variation_step)
        history3['train'].append(ce_loss(train_logits3, y_train)[0])
        history3['valid'].append(ce_loss(valid_logits3, y_valid)[0])
    if variation_step == 600:
        break
    variation_batch = variation_batch_rng.choice(len(X_train), 50, replace=False)
    variation_logits, variation_cache = cnn_forward(X_train[variation_batch], p3)
    variation_loss, variation_dlogits = ce_loss(variation_logits, y_train[variation_batch])
    _, variation_gradients = cnn_backward(variation_dlogits, variation_cache, p3)
    adam_update(p3, variation_gradients, state3, variation_step + 1)
assert np.all(np.isfinite(history3['train'])) and history3['train'][-1] < history3['train'][0]
variation = {
    'filters': filters, 'head_input_dimension': head_dimension,
    'parameter_count': sum(value.size for value in p3.values()),
    'steps': 600, 'batch_size': 50, 'train_samples': len(y_train), 'valid_samples': len(y_valid),
    'train_initial_ce': float(history3['train'][0]), 'train_final_ce': float(history3['train'][-1]),
    'valid_initial_ce': float(history3['valid'][0]), 'valid_final_ce': float(history3['valid'][-1]),
    'train_accuracy': float(np.mean(train_logits3.argmax(axis=1) == y_train)),
    'valid_accuracy': float(np.mean(valid_logits3.argmax(axis=1) == y_valid)),
    'parameter_shapes': {key: list(value.shape) for key, value in p3.items()},
    'history': {key: list(map(float, values)) for key, values in history3.items()},
    'elapsed_seconds': time.time() - variation_started}

rehearsal_metrics = {
    'hand_convolution_output': small_Y.tolist(),
    'changed_filter_output': changed_output.tolist(),
    'parameter_count_examples': {'dense': (64+1)*6*6*6, 'local_only': (9+1)*6*6*6, 'shared': (9+1)*6},
    'notebook_conv_gradient_errors': {key: float(value) for key, value in conv_errors.items()},
    'notebook_cnn_gradient_errors': {key: float(value) for key, value in cnn_errors.items()},
    'probe_minimum_absolute_preactivation': float(np.min(np.abs(probe_cache[0]))),
    'coverage': coverage.tolist(),
    'learner_conv_gradient_errors': {key: float(value) for key, value in learner_conv_errors.items()},
    'learner_pool_result': learner_pool_result,
    'callback_calls': candidate_calls,
    'wrong_overlap_rejected': wrong_overlap_rejected,
    'wrong_overlap_exception': wrong_overlap_error,
    'wrong_pool_rejected': wrong_pool_rejected,
    'wrong_pool_exception': wrong_pool_error,
    'checker_contract_probes': contract_probes,
    'pool_gradient_errors': pool_gradient_errors,
    'first_maximum_tie_gradient': max_tie_gradient.tolist(),
    'periodic_equivariance_max_abs_error': float(np.max(np.abs(moved - np.roll(base,1,axis=3)))),
    'baseline': baseline, 'filter_variation': variation,
    'train_label_counts': np.bincount(y_train).tolist(), 'valid_label_counts': np.bincount(y_valid).tolist()
}
import json
print('REHEARSAL_JSON ' + json.dumps(rehearsal_metrics))
'''
(out/'independent_exercises.py').write_text(extra)
notebook.cells.append(nbformat.v4.new_code_cell(extra))
start = time.time()
client = NotebookClient(notebook, timeout=180, kernel_name='python3', resources={'metadata': {'path': str(source.parent)}})
try:
    client.execute()
finally:
    nbformat.write(notebook, out/'executed.ipynb')
elapsed = time.time()-start
images,streams,errors=[],[],[]
for i,cell in enumerate(notebook.cells):
    for output in cell.get('outputs',[]):
        if output.output_type == 'stream':
            streams.append({'cell': i, 'name': output.get('name'), 'text': output.get('text','')})
        if output.output_type == 'error':
            errors.append({'cell': i, 'ename': output.get('ename'), 'evalue': output.get('evalue')})
        data=output.get('data',{})
        if 'image/png' in data:
            blob=base64.b64decode(data['image/png'])
            image_path=out/f'figure_{len(images)+1:02d}_cell_{i}.png'
            image_path.write_bytes(blob)
            with Image.open(io.BytesIO(blob)) as im:
                im.load()
                images.append({'number':len(images)+1,'cell':i,'path':str(image_path),'width':im.width,'height':im.height,'mode':im.mode,'format':im.format})
for stream in streams:
    for line in stream['text'].splitlines():
        if line.startswith('REHEARSAL_JSON '):
            metrics=json.loads(line.split(' ',1)[1])
report={'source':str(source),'sha256':hashlib.sha256(raw).hexdigest(),'elapsed_seconds':elapsed,
        'original_cells':original_cells,'original_code_cells':original_code_cells,
        'executed_code_cells':sum(c.cell_type=='code' and c.get('execution_count') is not None for c in notebook.cells),
        'source_changed_during_execution':source.read_bytes()!=raw,
        'next_notebook_exists':(source.parent/'06_rnn.ipynb').exists(),
        'figure_count':len(images),'figures':images,'errors':errors,
        'warnings':[s for s in streams if s['name']=='stderr'],'metrics':metrics}
(out/'execution.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
(out/'streams.json').write_text(json.dumps(streams,indent=2,ensure_ascii=False))
print(json.dumps(report,indent=2,ensure_ascii=False))
