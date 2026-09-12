import json
from pathlib import Path

metrics = {}

# 받는 노드 i의 상류 기여를 보내는 노드 j로 직접 흩어 더합니다.
# 본문 backward는 호출하지 않고, 파라미터 공유도 노드별 outer product로 합합니다.
def my_gnn_layer_backward(dH, cache, p):
    X,S,messages,H = cache
    local = dH*(1-H*H)
    dX = np.zeros_like(X)
    grads = {key:np.zeros_like(value) for key,value in p.items()}
    for batch in range(X.shape[0]):
        for receiver in range(X.shape[1]):
            dz = local[batch,receiver]
            grads['Ws'] += np.outer(X[batch,receiver],dz)
            grads['Wn'] += np.outer(messages[batch,receiver],dz)
            grads['b'] += dz
            dX[batch,receiver] += p['Ws']@dz
            neighbor_gradient = p['Wn']@dz
            for sender in range(X.shape[1]):
                dX[batch,sender] += S[batch,receiver,sender]*neighbor_gradient
    return dX,grads


own_errors = check_gnn_layer(my_gnn_layer_backward)
metrics['own_backward'] = {'errors':own_errors,'max_relative_error':max(own_errors.values())}

# 원문 directed 예제에 양쪽 연결이 모두 없는 4번 노드(0-based index 3)를 추가합니다.
isolation_A = np.zeros((1,4,4))
isolation_A[0,:3,:3] = A_hand[0]
isolation_X = np.concatenate([X_hand,np.array([[[-2.,3.]]])],axis=1)
isolation_p = {'Ws':np.array([[0.2,0.1],[-0.1,0.3]]),
               'Wn':np.array([[0.4,-0.2],[0.05,0.1]]),'b':np.array([0.1,-0.2])}
np.testing.assert_allclose((isolation_A@isolation_X)[0],[[8,12],[0,0],[1,2],[0,0]])
np.testing.assert_allclose((aggregation_matrix(isolation_A,'mean')@isolation_X)[0],[[4,6],[0,0],[1,2],[0,0]])
assert not np.allclose(isolation_A@isolation_X,isolation_A.swapaxes(-1,-2)@isolation_X)
isolation_records = {}
for mode in ['sum','mean']:
    ih,icache = gnn_layer_forward(isolation_X,isolation_A,isolation_p,mode)
    upstream = np.ones_like(ih)
    idx,igrads = my_gnn_layer_backward(upstream,icache,isolation_p)
    isolated_forward = np.tanh(isolation_X[:,3]@isolation_p['Ws']+isolation_p['b'])
    isolated_gradient = ((1-ih[:,3]**2)@isolation_p['Ws'].T)
    np.testing.assert_allclose(ih[:,3],isolated_forward)
    np.testing.assert_allclose(idx[:,3],isolated_gradient)
    numeric_X = numerical_gradient(lambda:float(gnn_layer_forward(isolation_X,isolation_A,isolation_p,mode)[0].sum()),isolation_X)
    isolation_error = rel_error(idx,numeric_X)
    assert isolation_error<2e-6
    isolation_records[mode] = {'all_input_gradient_error':isolation_error,
        'isolated_output':ih[:,3].tolist(),'isolated_input_gradient':idx[:,3].tolist(),
        'incoming_zero_but_sends_input_gradient':idx[:,1].tolist()}
metrics['direction_and_isolation'] = {'A':isolation_A.tolist(),
    'incoming_degree':isolation_A.sum(axis=-1).tolist(),
    'outgoing_degree':isolation_A.sum(axis=-2).tolist(),
    'true_isolated_node':3,'incoming_zero_but_connected_node':1,'records':isolation_records}

# 세 가지 독립 오답: 전치 누락, 평균을 합으로 취급, incoming=0에서 self까지 지우기.
def wrong_no_transpose(dH,cache,p):
    X,S,messages,H = cache
    _,grads = my_gnn_layer_backward(dH,cache,p)
    local = dH*(1-H**2)
    return local@p['Ws'].T+S@(local@p['Wn'].T),grads

def wrong_mean_as_sum(dH,cache,p):
    X,S,messages,H = cache
    unnormalized = (S!=0).astype(float)
    wrong_cache = (X,unnormalized,unnormalized@X,H)
    return my_gnn_layer_backward(dH,wrong_cache,p)

def wrong_drop_incoming_zero(dH,cache,p):
    X,S,messages,H = cache
    dX,grads = my_gnn_layer_backward(dH,cache,p)
    dX[S.sum(axis=-1)==0] = 0
    return dX,grads

def wrong_average_shared_grads(dH,cache,p):
    dX,grads = my_gnn_layer_backward(dH,cache,p)
    return dX,{key:value/cache[0].shape[1] for key,value in grads.items()}

trials = []
for name,function in [('notebook_wrong_direction',wrong_neighbor_direction),
                      ('no_transpose',wrong_no_transpose),('mean_as_sum',wrong_mean_as_sum),
                      ('incoming_zero_erases_self',wrong_drop_incoming_zero),
                      ('extra_node_average',wrong_average_shared_grads)]:
    try:
        check_gnn_layer(function)
    except AssertionError as error:
        trials.append({'name':name,'detected':True,'error':str(error)})
    else:
        trials.append({'name':name,'detected':False})
assert all(item['detected'] for item in trials)
assert trials[2]['error'].startswith("('mean',")
metrics['intentional_errors'] = trials

# 그래프별로 서로 다른 순열을 적용하고, 직접 backward로 전체 미분도 대조합니다.
permutation_rng = np.random.default_rng(1010)
permutations = np.stack([permutation_rng.permutation(6) for _ in range(3)])
perm_X = np.stack([X_valid[i,permutations[i]] for i in range(3)])
perm_A = np.stack([A_valid[i][permutations[i]][:,permutations[i]] for i in range(3)])
permutation_records = {}
for mode in ['sum','mean']:
    ordinary_H,ordinary_cache = gnn_layer_forward(X_valid[:3],A_valid[:3],layers[0],mode)
    perm_H,perm_cache = gnn_layer_forward(perm_X,perm_A,layers[0],mode)
    expected_H = np.stack([ordinary_H[i,permutations[i]] for i in range(3)])
    np.testing.assert_allclose(perm_H,expected_H,atol=1e-12)
    upstream = permutation_rng.normal(size=ordinary_H.shape)
    perm_upstream = np.stack([upstream[i,permutations[i]] for i in range(3)])
    ordinary_dx,ordinary_dp = my_gnn_layer_backward(upstream,ordinary_cache,layers[0])
    perm_dx,perm_dp = my_gnn_layer_backward(perm_upstream,perm_cache,layers[0])
    expected_dx = np.stack([ordinary_dx[i,permutations[i]] for i in range(3)])
    np.testing.assert_allclose(perm_dx,expected_dx,atol=1e-12)
    for key in ordinary_dp:np.testing.assert_allclose(perm_dp[key],ordinary_dp[key],atol=1e-12)
    ordinary_logits,ordinary_graph_cache = graph_forward(X_valid[:3],A_valid[:3],layers,head,mode)
    perm_logits,perm_graph_cache = graph_forward(perm_X,perm_A,layers,head,mode)
    np.testing.assert_allclose(perm_logits,ordinary_logits,atol=1e-12)
    graph_upstream = permutation_rng.normal(size=ordinary_logits.shape)
    gx,gparams,ghead = graph_backward(graph_upstream,ordinary_graph_cache,layers,head,
                                     layer_backward_fn=my_gnn_layer_backward)
    gxp,gparamsp,gheadp = graph_backward(graph_upstream,perm_graph_cache,layers,head,
                                       layer_backward_fn=my_gnn_layer_backward)
    expected_gx = np.stack([gx[i,permutations[i]] for i in range(3)])
    np.testing.assert_allclose(gxp,expected_gx,atol=1e-12)
    graph_param_errors = []
    for first,second in zip([*gparams,ghead],[*gparamsp,gheadp]):
        for key in first:
            np.testing.assert_allclose(first[key],second[key],atol=1e-12)
            graph_param_errors.append(float(np.max(np.abs(first[key]-second[key]))))
    permutation_records[mode] = {
        'node_output_max_abs_error':float(np.max(np.abs(perm_H-expected_H))),
        'node_input_gradient_max_abs_error':float(np.max(np.abs(perm_dx-expected_dx))),
        'layer_shared_gradient_max_abs_error':max(float(np.max(np.abs(perm_dp[key]-ordinary_dp[key]))) for key in ordinary_dp),
        'graph_output_max_abs_error':float(np.max(np.abs(perm_logits-ordinary_logits))),
        'graph_input_gradient_max_abs_error':float(np.max(np.abs(gxp-expected_gx))),
        'graph_shared_gradient_max_abs_error':max(graph_param_errors),
    }
metrics['permutations'] = {'permutations':permutations.tolist(),'modes':permutation_records,
                         'backward_used':'my_gnn_layer_backward'}

# 2층·readout·head 전부에서도 직접 쓴 layer backward를 수치미분과 확인합니다.
full_logits,full_cache = graph_forward(probe_X,probe_A,probe_layers,probe_head)
full_upstream = np.random.default_rng(105).normal(size=full_logits.shape)
full_dx,full_dlayers,full_dhead = graph_backward(full_upstream,full_cache,probe_layers,probe_head,
                                               layer_backward_fn=my_gnn_layer_backward)
full_scalar = lambda:float(np.sum(graph_forward(probe_X,probe_A,probe_layers,probe_head)[0]*full_upstream))
full_errors = {'X':rel_error(full_dx,numerical_gradient(full_scalar,probe_X))}
for i,(params,grads) in enumerate(zip([*probe_layers,probe_head],[*full_dlayers,full_dhead])):
    for key,value in params.items():full_errors[f'{i}/{key}']=rel_error(grads[key],numerical_gradient(full_scalar,value))
assert max(full_errors.values())<2e-6
metrics['full_graph_backward']={'errors':full_errors,'max_relative_error':max(full_errors.values())}

# 노드 수 8: 학습하지 않고 기존 가중치로 예측하고, 같은 backward의 shape도 확인합니다.
snapshots = [{key:value.copy() for key,value in item.items()} for item in [*layers,head]]
eight_X,eight_A,eight_y = graph_data(64,8,109)
eight_logits,eight_cache = graph_forward(eight_X,eight_A,layers,head)
eight_loss,eight_dlogits = ce_loss(eight_logits,eight_y)
eight_dx,eight_dlayers,eight_dhead = graph_backward(eight_dlogits,eight_cache,layers,head,
                                                  layer_backward_fn=my_gnn_layer_backward)
assert eight_dx.shape==(64,8,1)
for params,grads,original in zip([*layers,head],[*eight_dlayers,eight_dhead],snapshots):
    for key in params:
        assert params[key].shape==grads[key].shape
        assert np.isfinite(grads[key]).all()
        np.testing.assert_array_equal(params[key],original[key])
metrics['six_node_baseline']={'train_count':len(y_train),'valid_count':len(y_valid),'updates':350,
    'train_ce_initial':float(history['train'][0]),'train_ce_final':float(history['train'][-1]),
    'valid_ce_final':float(history['valid'][-1]),'valid_accuracy':float(accuracy),
    'parameters':sum(value.size for item in [*layers,head] for value in item.values())}
metrics['eight_node_variation']={'additional_updates':0,'X_shape':list(eight_X.shape),
    'A_shape':list(eight_A.shape),'logits_shape':list(eight_logits.shape),'dX_shape':list(eight_dx.shape),
    'ce':float(eight_loss),'accuracy':float(np.mean(eight_logits.argmax(axis=-1)==eight_y)),
    'parameters':sum(value.size for item in [*layers,head] for value in item.values()),
    'parameter_shapes':[{key:list(value.shape) for key,value in item.items()} for item in [*layers,head]],
    'weights_unchanged':True,'backward_used':'my_gnn_layer_backward'}

# 두 regular graph가 실제로 연결성은 다르다는 것을 별도 탐색으로 확인합니다.
def connected_components(adjacency):
    unseen=set(range(len(adjacency)))
    components=[]
    while unseen:
        start=next(iter(unseen))
        reached={start}
        queue=[start]
        while queue:
            node=queue.pop()
            for other in np.flatnonzero(adjacency[node]):
                if int(other) not in reached:
                    reached.add(int(other))
                    queue.append(int(other))
        unseen-=reached
        components.append(sorted(reached))
    return components

cycle_components=connected_components(cycle6)
triangle_components=connected_components(triangles)
assert len(cycle_components)==1 and len(triangle_components)==2
np.testing.assert_array_equal(cycle6.sum(axis=1),np.full(6,2.))
np.testing.assert_array_equal(triangles.sum(axis=1),np.full(6,2.))

# 모든 노드가 같은 표현이고 이웃 수가 2이면 다음 층도 같다는 귀납을 6층까지 실행합니다.
limit_rng=np.random.default_rng(1011)
representation=np.ones((2,6,1))
regular_layer_differences=[]
for depth in range(1,7):
    din=representation.shape[-1]
    layer_p={'Ws':limit_rng.normal(size=(din,4))*0.3,
             'Wn':limit_rng.normal(size=(din,4))*0.2,'b':limit_rng.normal(size=4)*0.1}
    representation,_=gnn_layer_forward(representation,regular_pair,layer_p,'sum')
    np.testing.assert_allclose(representation[0],representation[1],atol=1e-12)
    np.testing.assert_allclose(representation,np.broadcast_to(representation[:,:1],representation.shape),atol=1e-12)
    regular_layer_differences.append(float(np.max(np.abs(representation[0]-representation[1]))))

mean_representation=np.ones((2,6,1))
mean_differences=[]
for depth in range(1,5):
    din=mean_representation.shape[-1]
    layer_p={'Ws':limit_rng.normal(size=(din,3))*0.3,
             'Wn':limit_rng.normal(size=(din,3))*0.2,'b':limit_rng.normal(size=3)*0.1}
    mean_representation,_=gnn_layer_forward(mean_representation,path_star,layer_p,'mean')
    np.testing.assert_allclose(mean_representation[0],mean_representation[1],atol=1e-12)
    mean_differences.append(float(np.max(np.abs(mean_representation[0]-mean_representation[1]))))
metrics['indistinguishable_graphs']={'cycle_components':cycle_components,'triangle_components':triangle_components,
    'degree_each_node':2,'regular_pair_six_layer_max_differences':regular_layer_differences,
    'mean_path_star_four_layer_max_differences':mean_differences,
    'trained_regular_pair_logit_gap':float(np.max(np.abs(regular_logits[0]-regular_logits[1]))),
    'trained_mean_path_star_logit_gap':float(np.max(np.abs(mean_logits[0]-mean_logits[1])))}

# 4절의 평균 degree만으로 선형 readout이 같다는 설명에 필요한 상수 특성 조건을 확인합니다.
constant_features=np.ones((6,1))
perturbed_features=constant_features.copy()
perturbed_features[0,0]+=0.03
linear_constant=[float((A@constant_features).mean()) for A in [path6,star6]]
linear_perturbed=[float((A@perturbed_features).mean()) for A in [path6,star6]]
np.testing.assert_allclose(linear_constant[0],linear_constant[1])
assert abs(linear_perturbed[0]-linear_perturbed[1])>0.01
metrics['linear_readout_condition_counterexample']={
    'cell':13,'equal_mean_degree':float(path6.sum(axis=1).mean()),
    'constant_feature_readouts':linear_constant,'shared_perturbed_features':perturbed_features.ravel().tolist(),
    'perturbed_feature_readouts':linear_perturbed,'gap':abs(linear_perturbed[0]-linear_perturbed[1])}

Path('/tmp/dlfs-architectures/rehearsal10/metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2))
print(json.dumps(metrics,ensure_ascii=False,indent=2))
