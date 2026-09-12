
# Independent backward: add every receiving node's derivative to its sender nodes.
learner_calls=0
def my_gnn_layer_backward(dH,cache,p):
    global learner_calls
    learner_calls+=1
    X,S,messages,H=cache
    B,M,D=X.shape
    width=H.shape[-1]
    dX=np.zeros_like(X)
    grads={'Ws':np.zeros((D,width)),'Wn':np.zeros((D,width)),'b':np.zeros(width)}
    for graph_index in range(B):
        for receiver in range(M):
            dz=dH[graph_index,receiver]*(1-H[graph_index,receiver]**2)
            grads['Ws']+=X[graph_index,receiver,:,None]*dz[None,:]
            grads['Wn']+=messages[graph_index,receiver,:,None]*dz[None,:]
            grads['b']+=dz
            dX[graph_index,receiver]+=p['Ws']@dz
            for sender in range(M):
                dX[graph_index,sender]+=S[graph_index,receiver,sender]*(p['Wn']@dz)
    return dX,grads
learner_layer_errors=check_gnn_layer(my_gnn_layer_backward)
assert learner_calls==2 and len(learner_layer_errors)==8

# Apply the delivered permutation test to the independently written candidate.
permutation_errors={}
for mode in ['sum','mean']:
    original_H,original_cache=gnn_layer_forward(X_valid[:3],A_valid[:3],layers[0],mode)
    permuted_H,permuted_cache=gnn_layer_forward(Xp,Ap,layers[0],mode)
    permutation_G=np.random.default_rng(108).normal(size=original_H.shape)
    original_dx,original_grads=my_gnn_layer_backward(permutation_G,original_cache,layers[0])
    permuted_dx,permuted_grads=my_gnn_layer_backward(permutation_G[:,permutation],permuted_cache,layers[0])
    original_logits,_=graph_forward(X_valid[:3],A_valid[:3],layers,head,mode)
    permuted_logits,_=graph_forward(Xp,Ap,layers,head,mode)
    errors={'node_output':float(np.max(np.abs(permuted_H-original_H[:,permutation]))),
            'input_gradient':float(np.max(np.abs(permuted_dx-original_dx[:,permutation]))),
            'graph_logits':float(np.max(np.abs(permuted_logits-original_logits)))}
    for key in original_grads:
        errors[key]=float(np.max(np.abs(permuted_grads[key]-original_grads[key])))
    assert max(errors.values())<1e-12
    permutation_errors[mode]=errors

# Compose both graph layers with the candidate and check every input/parameter.
learner_graph_errors={}
for graph_mode in ['sum','mean']:
    audit_logits,audit_cache=graph_forward(probe_X,probe_A,probe_layers,probe_head,graph_mode)
    audit_G=np.random.default_rng(105).normal(size=audit_logits.shape)
    audit_dx,audit_dlayers,audit_dhead=graph_backward(audit_G,audit_cache,probe_layers,probe_head,my_gnn_layer_backward)
    objective=lambda:float(np.sum(graph_forward(probe_X,probe_A,probe_layers,probe_head,graph_mode)[0]*audit_G))
    errors={'X':float(rel_error(audit_dx,numerical_gradient(objective,probe_X)))}
    for layer_index,(parameters,gradients) in enumerate(zip([*probe_layers,probe_head],[*audit_dlayers,audit_dhead])):
        for key,value in parameters.items():
            errors[f'{layer_index}/{key}']=float(rel_error(gradients[key],numerical_gradient(objective,value)))
    assert max(errors.values())<2e-6
    learner_graph_errors[graph_mode]=errors

# A directed graph distinguishes 'no incoming edges' from completely isolated.
isolated_A=np.array([[[0.,1.,1.,0.],[0.,0.,0.,0.],[1.,0.,0.,0.],[0.,0.,0.,0.]]])
isolated_X=np.array([[[0.2,-0.1],[0.3,0.4],[-0.2,0.5],[1.,-2.]]])
isolated_p={'Ws':np.array([[0.3,-0.2],[0.1,0.4]]),
            'Wn':np.array([[0.2,0.5],[-0.3,0.1]]),'b':np.array([0.05,-0.1])}
isolated_checks={}
for mode in ['sum','mean']:
    isolated_H,isolated_cache=gnn_layer_forward(isolated_X,isolated_A,isolated_p,mode)
    isolated_G=np.zeros_like(isolated_H)
    isolated_G[0,3]=[0.7,-0.2]
    isolated_dx,isolated_grads=my_gnn_layer_backward(isolated_G,isolated_cache,isolated_p)
    self_H=np.tanh(isolated_X[0,3]@isolated_p['Ws']+isolated_p['b'])
    self_dz=isolated_G[0,3]*(1-self_H**2)
    self_dx=isolated_p['Ws']@self_dz
    np.testing.assert_allclose(isolated_H[0,3],self_H)
    np.testing.assert_allclose(isolated_dx[0,3],self_dx)
    np.testing.assert_allclose(isolated_dx[0,:3],0)
    np.testing.assert_allclose(isolated_grads['Wn'],0)
    assert np.linalg.norm(isolated_dx[0,3])>0
    # Node 1 receives no messages but sends to node 0, so its input gets that derivative.
    receiving_G=np.zeros_like(isolated_H)
    receiving_G[0,0]=[0.7,-0.2]
    sender_dx,_=my_gnn_layer_backward(receiving_G,isolated_cache,isolated_p)
    expected_sender=isolated_cache[1][0,0,1]*(isolated_p['Wn']@(receiving_G[0,0]*(1-isolated_H[0,0]**2)))
    np.testing.assert_allclose(sender_dx[0,1],expected_sender)
    assert np.linalg.norm(sender_dx[0,1])>0
    np.testing.assert_allclose(sender_dx[0,3],0)
    isolated_checks[mode]={'zero_incoming_rows':np.flatnonzero(isolated_A[0].sum(axis=1)==0).tolist(),
        'zero_outgoing_columns':np.flatnonzero(isolated_A[0].sum(axis=0)==0).tolist(),
        'fully_isolated_output':isolated_H[0,3].tolist(),'fully_isolated_self_gradient':isolated_dx[0,3].tolist(),
        'fully_isolated_neighbor_weight_gradient_max':float(np.max(np.abs(isolated_grads['Wn']))),
        'no_incoming_but_outgoing_node_gradient':sender_dx[0,1].tolist()}

# Direct hand sums/means and asymmetric normalization of an undirected star.
np.testing.assert_allclose(aggregate_loops(X_hand,aggregation_matrix(A_hand,'mean')),mean_messages)
star_S=aggregation_matrix(star6[None],'mean')[0]
np.testing.assert_allclose(star_S[0,1:],0.2)
np.testing.assert_allclose(star_S[1:,0],1)
assert not np.array_equal(star_S,star_S.T)

# Probe incomplete outputs and the exact topology mistakes addressed by the lesson.
def probe_checker(candidate):
    try:
        result=check_gnn_layer(candidate)
    except Exception as exc:
        return {'rejected':True,'exception_type':type(exc).__name__,'exception':str(exc)}
    return {'rejected':False,'returned_errors':result}
def wrong_without_self(dH,cache,p):
    dX,gradients=my_gnn_layer_backward(dH,cache,p)
    X,S,messages,H=cache
    return dX-(dH*(1-H**2))@p['Ws'].T,gradients
def wrong_missing_neighbor_weight(dH,cache,p):
    dX,gradients=my_gnn_layer_backward(dH,cache,p)
    del gradients['Wn']
    return dX,gradients
def wrong_gradient_shape(dH,cache,p):
    dX,gradients=my_gnn_layer_backward(dH,cache,p)
    gradients['b']=gradients['b'][None,:]
    return dX,gradients
checker_probes={
    'wrong_neighbor_direction':probe_checker(wrong_neighbor_direction),
    'no_self_input_gradient':probe_checker(wrong_without_self),
    'empty_return':probe_checker(lambda dH,cache,p:()),
    'missing_neighbor_weight':probe_checker(wrong_missing_neighbor_weight),
    'wrong_parameter_gradient_shape':probe_checker(wrong_gradient_shape)}
assert all(value['rejected'] for value in checker_probes.values())

# Independently run the 8-node example without changing a parameter.
parameter_snapshot=[{key:value.copy() for key,value in parameters.items()} for parameters in [*layers,head]]
my_X8,my_A8,my_y8=graph_data(64,8,109)
my_logits8,my_cache8=graph_forward(my_X8,my_A8,layers,head)
my_loss8,my_dlogits8=ce_loss(my_logits8,my_y8)
my_dx8,my_dlayers8,my_dhead8=graph_backward(my_dlogits8,my_cache8,layers,head,my_gnn_layer_backward)
assert my_logits8.shape==(64,2) and my_dx8.shape==(64,8,1)
for before,after in zip(parameter_snapshot,[*layers,head]):
    for key in before:
        np.testing.assert_array_equal(before[key],after[key])
node_variation={'node_count':8,'samples':64,'ce':float(my_loss8),
    'accuracy':float(np.mean(my_logits8.argmax(axis=-1)==my_y8)),
    'logit_shape':list(my_logits8.shape),'input_gradient_shape':list(my_dx8.shape),
    'parameter_count':sum(value.size for parameters in [*layers,head] for value in parameters.values()),
    'parameter_unchanged':True,'extra_training_steps':0}

# Confirm that the stated expression limits persist with new parameters and more layers.
regular_depth_errors=[]
mean_depth_errors=[]
regular_features=np.ones((2,6,1))
mean_features=np.ones((2,6,1))
limit_rng=np.random.default_rng(111)
for depth in range(1,6):
    width=depth+2
    input_width=regular_features.shape[-1]
    limit_p={'Ws':limit_rng.normal(size=(input_width,width))*0.3,
             'Wn':limit_rng.normal(size=(input_width,width))*0.2,
             'b':limit_rng.normal(size=width)*0.1}
    regular_features,_=gnn_layer_forward(regular_features,regular_pair,limit_p,'sum')
    mean_features,_=gnn_layer_forward(mean_features,path_star,limit_p,'mean')
    regular_error=float(np.max(np.abs(regular_features[0]-regular_features[1])))
    mean_error=float(np.max(np.abs(mean_features[0]-mean_features[1])))
    np.testing.assert_allclose(regular_features[0],regular_features[1],atol=1e-12)
    np.testing.assert_allclose(mean_features[0],mean_features[1],atol=1e-12)
    regular_depth_errors.append(regular_error)
    mean_depth_errors.append(mean_error)

# Linear layer + readout loses the equal-average-degree distinction with constant X.
linear_Ws=np.array([[0.7,-0.2]])
linear_Wn=np.array([[0.3,0.5]])
linear_b=np.array([0.1,-0.4])
linear_nodes=constant_X@linear_Ws+(path_star@constant_X)@linear_Wn+linear_b
linear_readouts=linear_nodes.mean(axis=1)
np.testing.assert_allclose(linear_readouts[0],linear_readouts[1],atol=1e-12)
# Degree as a feature can expose this particular lost information to a nonlinear self path.
degree_features=path_star.sum(axis=-1)[...,None]
degree_p={'Ws':np.ones((1,1)),'Wn':np.zeros((1,1)),'b':np.zeros(1)}
degree_H,_=gnn_layer_forward(degree_features,path_star,degree_p,'mean')
degree_readouts=degree_H.mean(axis=1)
assert abs(float(degree_readouts[0,0]-degree_readouts[1,0]))>1e-3

baseline={'node_count':6,'train_count':128,'valid_count':64,'steps':350,
    'train_initial_ce':float(history['train'][0]),'train_final_ce':float(history['train'][-1]),
    'valid_initial_ce':float(history['valid'][0]),'valid_final_ce':float(history['valid'][-1]),
    'valid_accuracy':float(accuracy),'parameters':sum(value.size for parameters in [*layers,head] for value in parameters.values()),
    'history':{key:list(map(float,values)) for key,values in history.items()}}
rehearsal_metrics={
    'baseline':baseline,'eight_node_variation':node_variation,
    'hand_sum':sum_messages.tolist(),'hand_mean':mean_messages.tolist(),
    'notebook_layer_errors':gnn_layer_errors,'notebook_graph_errors':graph_errors,
    'learner_layer_errors':learner_layer_errors,'learner_graph_errors':learner_graph_errors,
    'permutation_errors':permutation_errors,'isolated_node_checks':isolated_checks,
    'checker_probes':checker_probes,'candidate_calls':learner_calls,
    'star_mean_matrix_max_asymmetry':float(np.max(np.abs(star_S-star_S.T))),
    'one_hop':one_hop.astype(int).tolist(),'two_hop':two_hop.astype(int).tolist(),
    'limitations':{'mean_path_star_logit_max_abs_difference':float(np.max(np.abs(mean_logits[0]-mean_logits[1]))),
        'sum_regular_graph_logit_max_abs_difference':float(np.max(np.abs(regular_logits[0]-regular_logits[1]))),
        'sum_regular_graph_node_differences_by_depth':regular_depth_errors,
        'mean_path_star_node_differences_by_depth':mean_depth_errors,
        'equal_average_degree':float(path6.sum(axis=1).mean()),
        'linear_readouts':linear_readouts.tolist(),
        'degree_feature_nonlinear_readouts':degree_readouts.tolist()}}
import json
print('REHEARSAL_JSON '+json.dumps(rehearsal_metrics))
