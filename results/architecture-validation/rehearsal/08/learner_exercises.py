import json
from pathlib import Path

metrics = {}

# query 하나씩 Jacobian을 명시적으로 만들어 미분합니다.
# J[i,j]=dp_i/ds_j이며 key 위치끼리의 상호작용을 모두 포함합니다.
def my_attention_backward(dO, cache):
    Q, K, V, P, mask, scale = cache
    dQ = np.zeros_like(Q)
    dK = np.zeros_like(K)
    dV = np.zeros_like(V)
    for sample in range(Q.shape[0]):
        for query in range(Q.shape[1]):
            probabilities = P[sample, query]
            probability_gradient = V[sample] @ dO[sample, query]
            jacobian = np.diag(probabilities) - np.outer(probabilities, probabilities)
            score_gradient = probability_gradient @ jacobian
            score_gradient[~mask[sample, query]] = 0
            dQ[sample, query] = score_gradient @ K[sample] / scale
            dK[sample] += np.outer(score_gradient, Q[sample, query]) / scale
            dV[sample] += np.outer(probabilities, dO[sample, query])
    return dQ, dK, dV


own_errors = check_attention_backward(my_attention_backward)
metrics['own_backward'] = {'errors':own_errors, 'max_relative_error':max(own_errors.values()),
    'Q_shape':[2,2,3], 'K_shape':[2,4,3], 'V_shape':[2,4,2], 'O_shape':[2,2,2]}

# 본문의 손계산 P=[3/4,1/4], L=sum(O)에서 backward를 수치로 펼칩니다.
hand_dO = np.ones_like(O_hand)
hand_dQ, hand_dK, hand_dV = my_attention_backward(hand_dO,hand_cache)
hand_dP = np.array([10.,20.])
hand_P = np.array([0.75,0.25])
hand_J = np.diag(hand_P)-np.outer(hand_P,hand_P)
hand_dS = hand_dP@hand_J
np.testing.assert_allclose(hand_dS,[-1.875,1.875])
np.testing.assert_allclose(hand_dQ,[[[-1.875/np.sqrt(2),1.875/np.sqrt(2)]]])
np.testing.assert_allclose(hand_dK,[[[-1.875*np.log(3),0.],[1.875*np.log(3),0.]]])
np.testing.assert_allclose(hand_dV,[[[0.75,0.75],[0.25,0.25]]])
np.testing.assert_allclose(hand_dS.sum(),0.,atol=1e-15)
metrics['hand_derivation'] = {'objective':'sum(O)','P':hand_P.tolist(),'softmax_jacobian':hand_J.tolist(),
    'dP':hand_dP.tolist(),'dS':hand_dS.tolist(),'dQ':hand_dQ.tolist(),
    'dK':hand_dK.tolist(),'dV':hand_dV.tolist()}

# key 수 3의 직사각형 attention 자체에서 다시 모든 입력을 수치미분합니다.
rect_rng = np.random.default_rng(808)
rect_Q = rect_rng.normal(size=(2,2,3))*0.3
rect_K = rect_rng.normal(size=(2,3,3))*0.3
rect_V = rect_rng.normal(size=(2,3,2))*0.3
rect_mask = np.array([[1,0,1],[0,1,1]],dtype=bool)
rect_O, rect_cache = attention_forward(rect_Q,rect_K,rect_V,rect_mask)
rect_G = rect_rng.normal(size=rect_O.shape)
rect_grads = my_attention_backward(rect_G,rect_cache)
rect_scalar = lambda: float(np.sum(attention_forward(rect_Q,rect_K,rect_V,rect_mask)[0]*rect_G))
rect_errors = {}
for name,value,analytic in zip(['Q','K','V'],[rect_Q,rect_K,rect_V],rect_grads):
    rect_errors[name] = rel_error(analytic,numerical_gradient(rect_scalar,value))
assert max(rect_errors.values())<2e-6
metrics['rectangular_three_keys'] = {'Q_shape':list(rect_Q.shape),'K_shape':list(rect_K.shape),
    'V_shape':list(rect_V.shape),'P_shape':list(rect_cache[3].shape),'O_shape':list(rect_O.shape),
    'errors':rect_errors,'max_relative_error':max(rect_errors.values())}

# padding: 모든 query에서 제외한 key의 입력 미분은 정확히 0이어야 합니다.
mask_rng = np.random.default_rng(809)
test_Q = mask_rng.normal(size=(2,4,3))
test_K = mask_rng.normal(size=(2,4,3))
test_V = mask_rng.normal(size=(2,4,2))
padding = np.array([True,False,True,False])[None,None,:]
padding_O,padding_cache = attention_forward(test_Q,test_K,test_V,padding)
np.testing.assert_allclose(padding_cache[3].sum(axis=-1),1.)
assert np.all(padding_cache[3][:,:,[1,3]]==0.)
_,padding_dK,padding_dV = my_attention_backward(np.ones_like(padding_O),padding_cache)
np.testing.assert_array_equal(padding_dK[:,[1,3]],0.)
np.testing.assert_array_equal(padding_dV[:,[1,3]],0.)

# causal과 padding을 결합하고, 미래 K/V 변경에 대한 과거 출력 불변을 확인합니다.
causal4 = np.arange(4)[None,:] <= np.arange(4)[:,None]
combined4 = causal4 & np.array([True,True,True,False])[None,:]
causal_O,causal_cache = attention_forward(test_Q,test_K,test_V,combined4)
changed_K,changed_V = test_K.copy(),test_V.copy()
changed_K[:,2:] += 100
changed_V[:,2:] -= 100
causal_changed,_ = attention_forward(test_Q,changed_K,changed_V,combined4)
np.testing.assert_allclose(causal_O[:,:2],causal_changed[:,:2],atol=1e-12)
np.testing.assert_allclose(causal_cache[3].sum(axis=-1),1.)
assert np.all(causal_cache[3][:,~combined4]==0.)
bad_row = combined4.copy()
bad_row[2] = False
try:
    attention_forward(test_Q,test_K,test_V,bad_row)
except ValueError as error:
    all_masked_message = str(error)
else:
    raise AssertionError('모두 금지된 query 행이 거부되지 않았습니다')
metrics['mask_variations'] = {'padding_mask_shape':list(padding.shape),
    'padding_forbidden_probability_max':float(np.max(np.abs(padding_cache[3][:,:,[1,3]]))),
    'padding_forbidden_dK_max':float(np.max(np.abs(padding_dK[:,[1,3]]))),
    'padding_forbidden_dV_max':float(np.max(np.abs(padding_dV[:,[1,3]]))),
    'combined_mask_shape':list(combined4.shape),
    'past_output_max_change':float(np.max(np.abs(causal_O[:,:2]-causal_changed[:,:2]))),
    'all_masked_row_rejected':True,'all_masked_message':all_masked_message}

# 의도적 오답: 합하는 축, mask 적용, Q/K 전치의 오류를 독립적으로 만듭니다.
def wrong_query_softmax_axis(dO,cache):
    Q,K,V,P,mask,scale = cache
    dP = dO@V.swapaxes(-1,-2)
    dS = P*(dP-(dP*P).sum(axis=-2,keepdims=True))
    return dS@K/scale,dS.swapaxes(-1,-2)@Q/scale,P.swapaxes(-1,-2)@dO

def wrong_mask_ignored(dO,cache):
    Q,K,V,P,mask,scale = cache
    _,unmasked_cache = attention_forward(Q,K,V)
    return my_attention_backward(dO,unmasked_cache)

def wrong_qk_transpose(dO,cache):
    Q,K,V,P,mask,scale = cache
    dP = dO@V.swapaxes(-1,-2)
    dS = P*(dP-(dP*P).sum(axis=-1,keepdims=True))
    # dK에서는 query와key 축을 뒤집어야 하는데 그대로 곱한 오류입니다.
    return dS@K/scale,dS@Q/scale,P.swapaxes(-1,-2)@dO

wrong_trials = []
for name,function in [('elementwise_softmax',wrong_elementwise_softmax),
                      ('query_softmax_axis',wrong_query_softmax_axis),
                      ('mask_ignored',wrong_mask_ignored),('qk_transpose',wrong_qk_transpose)]:
    try:
        check_attention_backward(function)
    except (AssertionError,ValueError) as error:
        wrong_trials.append({'name':name,'detected':True,'error_type':type(error).__name__,'error':str(error)})
    else:
        wrong_trials.append({'name':name,'detected':False})
assert all(item['detected'] for item in wrong_trials)
metrics['intentional_errors'] = wrong_trials

# checker의 일반 assert가 빈 메시지를 내므로, 같은 두 조건에서 수치오차도 보관합니다.
negative_rng = np.random.default_rng(82)
negative_errors = {}
for masked in [False,True]:
    nq = negative_rng.normal(size=(2,2,3))
    nk = negative_rng.normal(size=(2,4,3))
    nv = negative_rng.normal(size=(2,4,2))
    allowed = np.array([[1,1,0,0],[0,1,1,1]],dtype=bool) if masked else None
    no,ncache = attention_forward(nq,nk,nv,allowed)
    upstream = negative_rng.normal(size=no.shape)
    numerical_Q = numerical_gradient(lambda:float(np.sum(attention_forward(nq,nk,nv,allowed)[0]*upstream)),nq)
    for name,function in [('elementwise_softmax',wrong_elementwise_softmax),
                          ('query_softmax_axis',wrong_query_softmax_axis),('mask_ignored',wrong_mask_ignored)]:
        negative_errors[f'{name}/mask={masked}/Q'] = rel_error(function(upstream,ncache)[0],numerical_Q)
metrics['intentional_error_values'] = negative_errors

# key 수 3 변형: 정체 one-hot 차원=3, 비교 차원=3, 값 차원=2입니다.
# query를 여러 개 주면 labels=(N,Tq), logits=(N,Tq,2), CE는 N*Tq개 항목을 평균냅니다.
def my_retrieval_data(n,seed,nkeys=3,nqueries=1):
    generator = np.random.default_rng(seed)
    keys = np.empty((n,nkeys,nkeys))
    values = np.empty((n,nkeys,2))
    queries = np.empty((n,nqueries,nkeys))
    labels = np.empty((n,nqueries),dtype=int)
    for sample in range(n):
        order = generator.permutation(nkeys)
        bits = generator.integers(0,2,size=nkeys)
        requested = generator.integers(nkeys,size=nqueries)
        keys[sample] = np.eye(nkeys)[order]
        values[sample] = np.eye(2)[bits[order]]
        queries[sample] = np.eye(nkeys)[requested]
        labels[sample] = bits[requested]
    return queries,keys,values,labels[:,0] if nqueries==1 else labels

def my_init_retrieval(nkeys=3,seed=85):
    generator = np.random.default_rng(seed)
    return {'Wq':generator.normal(size=(nkeys,nkeys))*0.5,
            'Wk':generator.normal(size=(nkeys,nkeys))*0.5,
            'Wo':generator.normal(size=(2,2))*0.7,'b':np.zeros(2)}

def my_retrieval_loss(Xq,Xk,V,y,parameters):
    Q = Xq@parameters['Wq']
    K = Xk@parameters['Wk']
    O,cache = attention_forward(Q,K,V)
    logits = O@parameters['Wo']+parameters['b']
    loss,dz_rows = ce_loss(logits.reshape(-1,2),y.reshape(-1))
    dz = dz_rows.reshape(logits.shape)
    dO = dz@parameters['Wo'].T
    dQ,dK,dV = my_attention_backward(dO,cache)
    grads = {'Wo':O.reshape(-1,2).T@dz_rows,'b':dz_rows.sum(axis=0),
             'Wq':Xq.reshape(-1,Xq.shape[-1]).T@dQ.reshape(-1,dQ.shape[-1]),
             'Wk':Xk.reshape(-1,Xk.shape[-1]).T@dK.reshape(-1,dK.shape[-1])}
    input_grads = (dQ@parameters['Wq'].T,dK@parameters['Wk'].T,dV)
    return loss,logits,grads,cache[3],input_grads

# query를 2개로 늘린 경우에도 전체 모델 미분과 평균 분모를 확인합니다.
two_query_data = my_retrieval_data(2,810,nqueries=2)
two_query_p = my_init_retrieval()
two_loss,two_logits,two_grads,_,two_input_grads = my_retrieval_loss(*two_query_data,two_query_p)
two_scalar = lambda:float(my_retrieval_loss(*two_query_data,two_query_p)[0])
two_errors = {key:rel_error(two_grads[key],numerical_gradient(two_scalar,value))
              for key,value in two_query_p.items()}
for name,value,gradient in zip(['Xq','Xk','V'],two_query_data[:3],two_input_grads):
    two_errors[name] = rel_error(gradient,numerical_gradient(two_scalar,value))
assert max(two_errors.values())<2e-6
separate_losses = []
for sample in range(2):
    for query in range(2):
        separate_losses.append(ce_loss(two_logits[sample,query][None],np.array([two_query_data[-1][sample,query]]))[0])
np.testing.assert_allclose(two_loss,np.mean(separate_losses))
metrics['two_queries'] = {'Xq_shape':list(two_query_data[0].shape),'Xk_shape':list(two_query_data[1].shape),
    'V_shape':list(two_query_data[2].shape),'labels_shape':list(two_query_data[-1].shape),
    'logits_shape':list(two_logits.shape),'ce_mean_denominator':4,
    'parameter_shapes':{key:list(value.shape) for key,value in two_query_p.items()},
    'errors':two_errors,'max_relative_error':max(two_errors.values())}

metrics['four_key_baseline'] = {'keys':4,'updates':500,'parameters':sum(value.size for value in p.values()),
    'train_ce_initial':float(history['train'][0]),'train_ce_final':float(history['train'][-1]),
    'valid_ce_final':float(history['valid'][-1]),'valid_accuracy':float(accuracy)}
train3,valid3 = my_retrieval_data(256,86),my_retrieval_data(128,87)
p3 = my_init_retrieval()
state3 = {key:(np.zeros_like(value),np.zeros_like(value)) for key,value in p3.items()}
three_history = []
for step in range(501):
    loss3,logits3,grads3,_,_ = my_retrieval_loss(*train3,p3)
    if step%50==0:
        validation_loss3,validation_logits3,_,weights3,_ = my_retrieval_loss(*valid3,p3)
        three_history.append({'step':step,'train_ce':float(loss3),'valid_ce':float(validation_loss3)})
    if step==500:
        break
    adam_step(p3,grads3,state3,step+1)
accuracy3 = np.mean(validation_logits3.reshape(-1,2).argmax(axis=-1)==valid3[-1].reshape(-1))
assert accuracy3>0.95 and loss3<0.1
requested_ids = valid3[0][:,0].argmax(axis=-1)
key_ids = valid3[1].argmax(axis=-1)
most_read_ids = key_ids[np.arange(len(key_ids)),weights3[:,0].argmax(axis=-1)]
metrics['three_key_training'] = {'keys':3,'updates':500,'fresh_model_and_optimizer_state':True,
    'backward_used':'my_attention_backward','train_shapes':[list(value.shape) for value in train3],
    'parameter_shapes':{key:list(value.shape) for key,value in p3.items()},
    'parameters':sum(value.size for value in p3.values()),'valid_count':len(valid3[-1]),
    'train_ce_final':float(loss3),'valid_ce_final':float(validation_loss3),
    'valid_accuracy':float(accuracy3),'query_key_argmax_match_rate':float(np.mean(most_read_ids==requested_ids)),
    'history':three_history}

Path('/tmp/dlfs-architectures/rehearsal08/metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2))
print(json.dumps(metrics,ensure_ascii=False,indent=2))
