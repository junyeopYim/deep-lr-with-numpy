import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['MPLCONFIGDIR']='/tmp/dlfs-architectures/rehearsal09/matplotlib'
os.environ['IPYTHONDIR']='/tmp/dlfs-architectures/rehearsal09/ipython'
import sys,json,pathlib,hashlib,time,base64,io
sys.dont_write_bytecode=True
import nbformat
from nbclient import NotebookClient
from PIL import Image
root=pathlib.Path('/home/junyeop/projects/deep-learning-from-scratch')
out=pathlib.Path('/tmp/dlfs-architectures/rehearsal09')
source=root/'notebooks/01_아키텍처/09_transformer.ipynb'
raw=source.read_bytes()
(out/'source.ipynb').write_bytes(raw)
notebook=nbformat.reads(raw.decode(),as_version=4)
original_cells=len(notebook.cells)
original_code_cells=sum(c.cell_type=='code' for c in notebook.cells)
extra='''
# Independent LayerNorm derivative: follow centered -> variance -> inverse -> normalized.
# The cache contains normalized and inverse, so centered = normalized / inverse.
learner_calls = {'layernorm': 0, 'block': 0}
def my_layernorm_backward(dY, cache):
    learner_calls['layernorm'] += 1
    normalized, inverse, gamma = cache
    D = dY.shape[-1]
    centered = normalized / inverse
    dnormalized = dY * gamma
    dinverse = np.sum(dnormalized * centered, axis=-1, keepdims=True)
    dvariance = dinverse * (-0.5) * inverse**3
    dcentered = dnormalized * inverse + dvariance * (2 / D) * centered
    dX = dcentered - dcentered.mean(axis=-1, keepdims=True)
    shared_axes = tuple(range(dY.ndim - 1))
    dgamma = np.sum(dY * normalized, axis=shared_axes)
    dbeta = np.sum(dY, axis=shared_axes)
    return dX, dgamma, dbeta
learner_ln_errors = check_layernorm(my_layernorm_backward)

# Independently assemble the block by sending each residual's derivative down both paths.
def my_block_backward(dY, cache, p):
    learner_calls['block'] += 1
    ln1_cache, attention_cache, ln2_cache, ffn_cache = cache
    dB, parameter_grads = ffn_backward(dY, ffn_cache, p)
    branch_dU, dgamma2, dbeta2 = my_layernorm_backward(dB, ln2_cache)
    total_dU = dY.copy() + branch_dU
    dA, attention_grads = mha_backward(total_dU, attention_cache, p)
    branch_dX, dgamma1, dbeta1 = my_layernorm_backward(dA, ln1_cache)
    dX = total_dU.copy() + branch_dX
    parameter_grads.update(attention_grads)
    parameter_grads.update({'gamma1':dgamma1, 'beta1':dbeta1, 'gamma2':dgamma2, 'beta2':dbeta2})
    return dX, parameter_grads
learner_block_errors = check_block_backward(my_block_backward)
learner_model_errors = check_transformer(my_block_backward)
assert learner_calls['block'] == 2

# A two-feature hand calculation explicitly keeps nonzero epsilon and non-unit gamma.
hand_X = np.array([[[1.,3.]]])
hand_gamma, hand_beta = np.array([2.,3.]), np.array([-1.,0.5])
hand_G = np.array([[[1.,-2.]]])
hand_output, hand_cache = layernorm_forward(hand_X, hand_gamma, hand_beta, eps=0.07)
hand_gradients = my_layernorm_backward(hand_G, hand_cache)
hand_r = 1 / np.sqrt(1.07)
np.testing.assert_allclose(hand_output, [[[-2*hand_r-1, 3*hand_r+0.5]]])
np.testing.assert_allclose(hand_gradients[0], [[[4*0.07/(1.07**1.5), -4*0.07/(1.07**1.5)]]])
np.testing.assert_allclose(hand_gradients[1], [-hand_r,-2*hand_r])
np.testing.assert_allclose(hand_gradients[2], [1.,-2.])
np.testing.assert_allclose(np.mean(hand_cache[0]**2), 1/1.07)

# Checkers must reject incomplete implementations and common calculus/axis errors.
def probe_checker(checker, candidate):
    try:
        result = checker(candidate)
    except Exception as exc:
        return {'rejected':True, 'exception_type':type(exc).__name__, 'exception':str(exc)}
    return {'rejected':False, 'returned_errors':result}

def wrong_ln_unit_variance(dY, cache):
    normalized, inverse, gamma = cache
    normalized_variance = np.mean(normalized**2, axis=-1, keepdims=True)
    unit_normalized = normalized / np.sqrt(normalized_variance)
    inverse_without_eps = inverse / np.sqrt(normalized_variance)
    u = dY * gamma
    dX = inverse_without_eps * (u-u.mean(axis=-1,keepdims=True)-unit_normalized*(u*unit_normalized).mean(axis=-1,keepdims=True))
    _, dg, db = my_layernorm_backward(dY, cache)
    return dX, dg, db

def wrong_ln_token_axis(dY, cache):
    normalized, inverse, gamma = cache
    u = dY * gamma
    dX = inverse * (u-u.mean(axis=1,keepdims=True)-normalized*(u*normalized).mean(axis=1,keepdims=True))
    _, dg, db = my_layernorm_backward(dY, cache)
    return dX, dg, db

def wrong_ln_parameter_keepdims(dY, cache):
    dX, dg, db = my_layernorm_backward(dY, cache)
    return dX, dg[None,None,:], db[None,None,:]

def wrong_block_gradient_shapes(dY, cache, p):
    dX, gradients = my_block_backward(dY, cache, p)
    return dX, {name:value[None] for name,value in gradients.items()}

def wrong_block_missing_key(dY, cache, p):
    dX, gradients = my_block_backward(dY, cache, p)
    del gradients['Wq']
    return dX, gradients

checker_probes = {
    'ln_empty':probe_checker(check_layernorm,lambda dY,cache:()),
    'ln_only_dx':probe_checker(check_layernorm,lambda dY,cache:(my_layernorm_backward(dY,cache)[0],)),
    'ln_ignore_epsilon':probe_checker(check_layernorm,wrong_ln_unit_variance),
    'ln_wrong_token_axis':probe_checker(check_layernorm,wrong_ln_token_axis),
    'ln_wrong_parameter_keepdims':probe_checker(check_layernorm,wrong_ln_parameter_keepdims),
    'block_missing_residual':probe_checker(check_block_backward,wrong_missing_residual),
    'model_missing_residual':probe_checker(check_transformer,wrong_missing_residual),
    'block_missing_parameter':probe_checker(check_block_backward,wrong_block_missing_key),
    'block_wrong_gradient_shapes':probe_checker(check_block_backward,wrong_block_gradient_shapes),
    'model_wrong_gradient_shapes':probe_checker(check_transformer,wrong_block_gradient_shapes)}
for key in ['ln_empty','ln_only_dx','ln_ignore_epsilon','ln_wrong_token_axis','block_missing_residual','model_missing_residual','block_missing_parameter']:
    assert checker_probes[key]['rejected'], key
try:
    update_gamma = np.ones(4)
    update_X = np.random.default_rng(991).normal(size=(2,3,4))
    _, update_cache = layernorm_forward(update_X, update_gamma, np.zeros(4))
    _, bad_dgamma, _ = wrong_ln_parameter_keepdims(np.ones_like(update_X), update_cache)
    update_gamma -= 0.01 * bad_dgamma
except ValueError as exc:
    wrong_shape_update = {'failed':True,'exception':str(exc),'parameter_shape':[4],'gradient_shape':[1,1,4]}
else:
    wrong_shape_update = {'failed':False}
assert wrong_shape_update['failed']

# Repeated embedding indices: reconstruct each row's contribution with explicit loops.
repeat_ids = np.array([[1,2,1],[0,1,2]])
repeat_contributions = np.array([[[1.,-2.],[3.,4.],[5.,6.]],[[-1.,2.],[7.,8.],[9.,-3.]]])
manual_embedding = np.zeros((3,2))
for n in range(repeat_ids.shape[0]):
    for t in range(repeat_ids.shape[1]):
        manual_embedding[repeat_ids[n,t]] += repeat_contributions[n,t]
np.testing.assert_array_equal(manual_embedding,[[-1.,2.],[13.,12.],[12.,1.]])
add_at_embedding = np.zeros_like(manual_embedding)
np.add.at(add_at_embedding,repeat_ids,repeat_contributions)
np.testing.assert_array_equal(add_at_embedding,manual_embedding)
fancy_embedding = np.zeros_like(manual_embedding)
fancy_embedding[repeat_ids] += repeat_contributions
assert not np.array_equal(fancy_embedding,manual_embedding)

# Both requested alternative head counts preserve split/merge; check MHA derivatives.
head_variant_errors = {}
for alternative_heads in [1,4]:
    np.testing.assert_array_equal(merge_heads(split_heads(head_X,alternative_heads)),head_X)
    alt_y, alt_cache = mha_forward(head_X,head_p,alternative_heads,causal3)
    alt_dx, alt_grads = mha_backward(head_G,alt_cache,head_p)
    objective = lambda:float(np.sum(mha_forward(head_X,head_p,alternative_heads,causal3)[0]*head_G))
    alt_errors = {key:float(rel_error(alt_grads[key],numerical_gradient(objective,value))) for key,value in head_p.items()}
    alt_errors['X'] = float(rel_error(alt_dx,numerical_gradient(objective,head_X)))
    assert max(alt_errors.values()) < 2e-6
    head_variant_errors[str(alternative_heads)] = alt_errors
invalid_heads_rejected = False
try:
    split_heads(np.zeros((1,4,12)),5)
except AssertionError:
    invalid_heads_rejected = True
assert invalid_heads_rejected

# Position/no-position and causal computations from the delivered experiment.
position_metrics = {
    'sinusoidal_first_row':position_grid[0].tolist(),
    'no_position_permutation_max_abs_error':float(np.max(np.abs(reordered_logits-no_pos_logits[:,permutation]))),
    'positions_removed_token_accuracy':float(np.mean(no_pos_logits.argmax(axis=-1)==valid_targets)),
    'causal_future_change_max_abs_error':float(np.max(np.abs(past1[:,:2]-past2[:,:2])))}
np.testing.assert_allclose(position_grid[:,0::2]**2+position_grid[:,1::2]**2,1,atol=1e-12)
np.testing.assert_allclose(head_cache[2][3].sum(axis=-1),1)
assert np.all(head_cache[2][3][~head_cache[2][4]]==0)
all_masked_rejected=False
try:
    mha_forward(head_X,head_p,2,np.zeros((3,3),dtype=bool))
except ValueError:
    all_masked_rejected=True
assert all_masked_rejected

# Masked future continuous inputs cannot receive a prefix-only objective gradient.
causal_block_p = init_block(4,6,993)
causal_block_X = np.random.default_rng(994).normal(size=(2,3,4))
causal_block_Y,causal_block_cache=block_forward(causal_block_X,causal_block_p,2,causal3)
prefix_G=np.zeros_like(causal_block_Y)
prefix_G[:,:1]=np.random.default_rng(995).normal(size=(2,1,4))
prefix_dx,_=my_block_backward(prefix_G,causal_block_cache,causal_block_p)
np.testing.assert_allclose(prefix_dx[:,1:],0,atol=1e-12)

# A batch-specific mask needs an explicit head axis. N==heads can conceal its omission.
mask_Q=np.zeros((2,2,3,2))
mask_K=np.zeros((2,2,3,2))
mask_V=np.arange(24.,dtype=float).reshape(2,2,3,2)
batch_allowed=np.broadcast_to(np.array([[True,True,False],[True,False,True]])[:,None,:],(2,3,3))
wrong_mask_O,wrong_mask_cache=attention_forward(mask_Q,mask_K,mask_V,batch_allowed)
correct_mask_O,correct_mask_cache=attention_forward(mask_Q,mask_K,mask_V,batch_allowed[:,None,:,:])
correct_full_mask=np.broadcast_to(batch_allowed[:,None,:,:],(2,2,3,3))
mask_metrics={
    'causal_probability_row_sum_max_error':float(np.max(np.abs(head_cache[2][3].sum(axis=-1)-1))),
    'causal_forbidden_probability_max':float(np.max(head_cache[2][3][~head_cache[2][4]])),
    'all_masked_rejected':all_masked_rejected,
    'future_input_gradient_max_abs':float(np.max(np.abs(prefix_dx[:,1:]))),
    'batch_mask_shape_without_head_axis':list(batch_allowed.shape),
    'batch_mask_shape_with_head_axis':list(batch_allowed[:,None,:,:].shape),
    'wrong_broadcast_forbidden_total_mass':float(np.sum(wrong_mask_cache[3][~correct_full_mask])),
    'correct_broadcast_forbidden_total_mass':float(np.sum(correct_mask_cache[3][~correct_full_mask])),
    'wrong_broadcast_output_max_abs_difference':float(np.max(np.abs(wrong_mask_O-correct_mask_O)))}

# Keep the baseline intact, then train the same D=12 model with four heads.
baseline={
    'heads':2,'head_width':6,'score_divisor':float(np.sqrt(6)),
    'parameters':sum(v.size for v in p.values()),'steps':800,'batch_size':64,
    'train_count':len(train_tokens),'valid_count':len(valid_tokens),
    'train_initial_ce':float(history['train'][0]),'train_final_ce':float(history['train'][-1]),
    'valid_initial_ce':float(history['valid'][0]),'valid_final_ce':float(history['valid'][-1]),
    'valid_token_accuracy':float(token_accuracy),'valid_sequence_accuracy':float(sequence_accuracy),
    'history':{k:list(map(float,v)) for k,v in history.items()}}
import time
variation_start=time.time()
p4=init_transformer()
state4={k:(np.zeros_like(v),np.zeros_like(v)) for k,v in p4.items()}
assert sum(v.size for v in p4.values())==baseline['parameters']
variation_rng=np.random.default_rng(902)
history4={'step':[],'train':[],'valid':[]}
for variation_step in range(801):
    if variation_step%80==0:
        train_logits4,_=transformer_forward(train_tokens,p4,heads=4)
        valid_logits4,_=transformer_forward(valid_tokens,p4,heads=4)
        history4['step'].append(variation_step)
        history4['train'].append(token_ce(train_logits4,train_targets)[0])
        history4['valid'].append(token_ce(valid_logits4,valid_targets)[0])
    if variation_step==800:
        break
    variation_batch=variation_rng.choice(len(train_tokens),64,replace=False)
    variation_logits,variation_cache=transformer_forward(train_tokens[variation_batch],p4,heads=4)
    _,variation_dlogits=token_ce(variation_logits,train_targets[variation_batch])
    variation_grads=transformer_backward(variation_dlogits,variation_cache,p4)
    variation_norm=np.sqrt(sum(np.sum(g*g) for g in variation_grads.values()))
    variation_scale=min(1.,1./(variation_norm+1e-12))
    adam_step(p4,{k:g*variation_scale for k,g in variation_grads.items()},state4,variation_step+1)
variation_predictions=valid_logits4.argmax(axis=-1)
assert np.all(np.isfinite(history4['train'])) and history4['train'][-1]<history4['train'][0]
variation={
    'heads':4,'head_width':3,'score_divisor':float(np.sqrt(3)),
    'parameters':sum(v.size for v in p4.values()),'steps':800,'batch_size':64,
    'train_count':len(train_tokens),'valid_count':len(valid_tokens),
    'train_initial_ce':float(history4['train'][0]),'train_final_ce':float(history4['train'][-1]),
    'valid_initial_ce':float(history4['valid'][0]),'valid_final_ce':float(history4['valid'][-1]),
    'valid_token_accuracy':float(np.mean(variation_predictions==valid_targets)),
    'valid_sequence_accuracy':float(np.mean(np.all(variation_predictions==valid_targets,axis=1))),
    'history':{k:list(map(float,v)) for k,v in history4.items()},
    'elapsed_seconds':time.time()-variation_start}

rehearsal_metrics={
    'layernorm_gradient_errors':ln_errors,'mha_gradient_errors':mha_errors,
    'block_gradient_errors':block_errors,'model_gradient_errors':transformer_errors,
    'learner_layernorm_errors':learner_ln_errors,'learner_block_errors':learner_block_errors,
    'learner_model_errors':learner_model_errors,'learner_calls':learner_calls,
    'layernorm_hand_output':hand_output.tolist(),'layernorm_hand_gradients':[v.tolist() for v in hand_gradients],
    'layernorm_hand_normalized_variance':float(np.mean(hand_cache[0]**2)),
    'checker_probes':checker_probes,'wrong_shape_parameter_update':wrong_shape_update,
    'manual_repeated_embedding':manual_embedding.tolist(),'add_at_embedding':add_at_embedding.tolist(),
    'wrong_fancy_index_embedding':fancy_embedding.tolist(),
    'head_variant_gradient_errors':head_variant_errors,'invalid_head_count_rejected':invalid_heads_rejected,
    'position_metrics':position_metrics,'mask_metrics':mask_metrics,
    'baseline':baseline,'four_head_variation':variation}
import json
print('REHEARSAL_JSON '+json.dumps(rehearsal_metrics))
'''
(out/'independent_exercises.py').write_text(extra)
notebook.cells.append(nbformat.v4.new_code_cell(extra))
start=time.time()
client=NotebookClient(notebook,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(source.parent)}})
try:
    client.execute()
finally:
    nbformat.write(notebook,out/'executed.ipynb')
elapsed=time.time()-start
images,streams,errors=[],[],[]
for i,cell in enumerate(notebook.cells):
    for output in cell.get('outputs',[]):
        if output.output_type=='stream':
            streams.append({'cell':i,'name':output.get('name'),'text':output.get('text','')})
        if output.output_type=='error':
            errors.append({'cell':i,'ename':output.get('ename'),'evalue':output.get('evalue')})
        data=output.get('data',{})
        if 'image/png' in data:
            blob=base64.b64decode(data['image/png'])
            path=out/f'figure_{len(images)+1:02d}_cell_{i}.png'
            path.write_bytes(blob)
            with Image.open(io.BytesIO(blob)) as im:
                im.load()
                images.append({'number':len(images)+1,'cell':i,'path':str(path),'width':im.width,'height':im.height,'mode':im.mode,'format':im.format})
for stream in streams:
    for line in stream['text'].splitlines():
        if line.startswith('REHEARSAL_JSON '):metrics=json.loads(line.split(' ',1)[1])
report={'source':str(source),'sha256':hashlib.sha256(raw).hexdigest(),'elapsed_seconds':elapsed,
        'original_cells':original_cells,'original_code_cells':original_code_cells,
        'executed_code_cells':sum(c.cell_type=='code' and c.get('execution_count') is not None for c in notebook.cells),
        'source_changed_during_execution':source.read_bytes()!=raw,
        'next_notebook_exists':(source.parent/'10_gnn.ipynb').exists(),
        'figure_count':len(images),'figures':images,'errors':errors,
        'warnings':[s for s in streams if s['name']=='stderr'],'metrics':metrics}
(out/'execution.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
(out/'streams.json').write_text(json.dumps(streams,indent=2,ensure_ascii=False))
print(json.dumps(report,indent=2,ensure_ascii=False))
