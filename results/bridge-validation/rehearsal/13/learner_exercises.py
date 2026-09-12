import json
from pathlib import Path

metrics = {}
my_stats = {'builds':0,'pullbacks':0}

# 출력 원소 y[n,k]=sum_d a[n,d]*b[d,k]를 직접 반복하며 두 부모 기여를 더합니다.
def my_matmul(a,b):
    if a.data.ndim!=2 or b.data.ndim!=2:
        raise ValueError('두 입력은 2차원이어야 합니다.')
    my_stats['builds']+=1
    def back(g):
        my_stats['pullbacks']+=1
        for n in range(a.data.shape[0]):
            for d in range(a.data.shape[1]):
                for k in range(b.data.shape[1]):
                    a.grad[n,d]+=g[n,k]*b.data[d,k]
                    b.grad[d,k]+=a.data[n,d]*g[n,k]
    return Node(a.data@b.data,(a,b),back)

metrics['candidate_checker'] = check_matmul(my_matmul)

# 서로 다른 연산에서 같은 입력을 사용하는 경우와 매 backward의 초기화입니다.
shared_x=Node(3.)
shared_graph=shared_x*shared_x+shared_x*shared_x
shared_graph.backward()
engine_first=float(shared_x.grad)
shared_graph.backward()
engine_second=float(shared_x.grad)
np.testing.assert_allclose([engine_first,engine_second],[12.,12.])
torch_shared=torch.tensor(3.,dtype=torch.float64,requires_grad=True)
(torch_shared*torch_shared+torch_shared*torch_shared).backward()
torch_first=float(torch_shared.grad)
(torch_shared*torch_shared+torch_shared*torch_shared).backward()
torch_second=float(torch_shared.grad)
torch_shared.grad=None
(torch_shared*torch_shared+torch_shared*torch_shared).backward()
torch_reset=float(torch_shared.grad)
np.testing.assert_allclose([torch_first,torch_second,torch_reset],[12.,24.,12.])
metrics['shared_and_reset']={'engine_first':engine_first,'engine_second':engine_second,
    'torch_first':torch_first,'torch_second_without_reset':torch_second,'torch_after_reset':torch_reset}

# 양쪽에서 서로 다른 축을 broadcasting하는 3차원 예제입니다.
broadcast_a=Node(np.arange(1.,7.).reshape(2,1,3))
broadcast_b=Node(np.arange(1.,5.).reshape(1,4,1))
broadcast_c=Node(0.)
broadcast_graph=(broadcast_a*broadcast_b+broadcast_b+broadcast_c).sum()
broadcast_graph.backward()
np.testing.assert_allclose(broadcast_a.grad,np.full((2,1,3),10.))
np.testing.assert_allclose(broadcast_b.grad,np.full((1,4,1),27.))
np.testing.assert_allclose(broadcast_c.grad,24.)
metrics['broadcasting']={'output_shape':[2,4,3],'a_shape':list(broadcast_a.data.shape),
    'b_shape':list(broadcast_b.data.shape),'a_gradient':broadcast_a.grad.tolist(),
    'b_gradient':broadcast_b.grad.tolist(),'scalar_gradient':float(broadcast_c.grad)}

# 벡터 출력에는 seed가 있어야 하며 shape도 맞아야 합니다.
seed_errors=[]
for seed in [None,np.ones(3)]:
    probe_vector=Node(np.array([2.,3.]))
    try:
        (probe_vector*probe_vector).backward(seed)
    except ValueError as error:
        seed_errors.append(str(error))
    else:
        raise AssertionError('잘못된 벡터 seed를 거부하지 않았습니다.')
metrics['invalid_seeds_rejected']=seed_errors

# stop-gradient는 원래 노드까지 돌아가는 경로를 끊고 현재 값은 유지합니다.
stop_live=Node(np.array([2.,-3.]))
full_stop_loss=(stop_live*stop_live).sum()
full_stop_loss.backward()
live_grad=stop_live.grad.copy()
frozen=Node(stop_live.data)
stopped_loss=(stop_live*frozen).sum()
stopped_loss.backward()
stopped_grad=stop_live.grad.copy()
torch_stop=torch.tensor([2.,-3.],dtype=torch.float64,requires_grad=True)
torch_stopped_loss=(torch_stop*torch_stop.detach()).sum()
torch_stopped_loss.backward()
np.testing.assert_allclose(live_grad,[4.,-6.])
np.testing.assert_allclose(stopped_grad,[2.,-3.])
np.testing.assert_allclose(stopped_grad,torch_stop.grad.numpy())
np.testing.assert_allclose([full_stop_loss.data,stopped_loss.data,torch_stopped_loss.item()],[13.,13.,13.])
metrics['stop_gradient']={'value':13.,'full_gradient':live_grad.tolist(),
    'stopped_gradient':stopped_grad.tolist(),'torch_gradient':torch_stop.grad.numpy().tolist()}

# 원본에서 사용한 같은 arrays와 입력으로 직접 후보·수식·중심차분·PyTorch를 비교합니다.
own_X=Node(X_check)
own_parameters=[Node(value) for value in arrays]
own_loss=graph_loss(own_X,Y_check,own_parameters,matmul_fn=my_matmul)
own_loss.backward()
manual_loss_value,manual_grads=manual_mlp(X_check,Y_check,arrays)
same_torch_parameters=[torch.tensor(value,dtype=torch.float64,requires_grad=True) for value in arrays]
same_torch_X=torch.tensor(X_check,dtype=torch.float64,requires_grad=True)
same_torch_Y=torch.tensor(Y_check,dtype=torch.float64)
pw1,pb1,pw2,pb2=same_torch_parameters
same_torch_prediction=torch.tanh(same_torch_X@pw1+pb1)@pw2+pb2
same_torch_loss=.5*((same_torch_prediction-same_torch_Y)**2).mean()
same_torch_loss.backward()
np.testing.assert_allclose([own_loss.data,same_torch_loss.item()],[manual_loss_value,manual_loss_value],rtol=1e-12)
comparison_errors={}
for name,node,reference,value,tvalue in zip(['W1','b1','W2','b2'],own_parameters,manual_grads,arrays,same_torch_parameters):
    numeric=numerical_gradient(lambda:manual_mlp(X_check,Y_check,arrays)[0],value)
    comparison_errors[name]={'manual':rel_error(node.grad,reference),
        'numeric':rel_error(node.grad,numeric),'torch':rel_error(node.grad,tvalue.grad.numpy())}
comparison_errors['X']={'torch':rel_error(own_X.grad,same_torch_X.grad.numpy()),
    'numeric':rel_error(own_X.grad,numerical_gradient(lambda:manual_mlp(X_check,Y_check,arrays)[0],X_check))}
assert max(item['torch'] for item in comparison_errors.values())<1e-12
assert max(item['numeric'] for item in comparison_errors.values())<1e-7
expected=[value-.07*gradient for value,gradient in zip(arrays,manual_grads)]
with torch.no_grad():
    for node,tvalue,target in zip(own_parameters,same_torch_parameters,expected):
        node.data-=.07*node.grad
        tvalue-=.07*tvalue.grad
        np.testing.assert_allclose(node.data,target,atol=1e-14)
        np.testing.assert_allclose(tvalue.detach().numpy(),target,atol=1e-14)
metrics['same_parameters_comparison']={'loss':float(own_loss.data),'torch_version':torch.__version__,
    'device':'cpu','dtype':'float64','parameter_shapes':[list(value.shape) for value in arrays],
    'errors':comparison_errors,'sgd_lr':.07,'all_four_sgd_updates_match':True}

# 안내대로 직접 후보를 넣어 너비 6을 실제 학습합니다.
stats_before=my_stats.copy()
own_trained,own_history=fit_graph_mlp(X_train,Y_train,hidden_size=6,matmul_fn=my_matmul)
training_calls={key:my_stats[key]-stats_before[key] for key in my_stats}
assert training_calls=={'builds':3602,'pullbacks':3600}
own_valid_prediction=graph_prediction(X_valid,own_trained,matmul_fn=my_matmul).data
own_valid_loss=float(.5*np.mean((own_valid_prediction-Y_valid)**2))
np.testing.assert_allclose(own_history,smaller_history,atol=1e-10,rtol=1e-9)
np.testing.assert_allclose(own_valid_loss,smaller_valid,atol=1e-10,rtol=1e-9)
assert own_history[-1]<.05*own_history[0] and own_valid_loss<.02
metrics['width_training']={'steps':1800,'lr':.1,'seed':131,
    'width12':{'train_initial':float(train_history[0]),'train_final':float(train_history[-1]),'valid':float(valid_half_mse)},
    'own_width6':{'train_initial':float(own_history[0]),'train_final':float(own_history[-1]),'valid':own_valid_loss,
        'parameter_shapes':[list(parameter.data.shape) for parameter in own_trained],
        'matmul_calls_during_fit':training_calls,'max_history_difference_from_reference':float(np.max(np.abs(own_history-smaller_history)))}}

# 오답의 검출 범위를 실제 후보 함수로 확인합니다.
def wrong_overwrite(a,b):
    def back(g):
        a.grad=g@b.data.T
        b.grad=a.data.T@g
    return Node(a.data@b.data,(a,b),back)

def wrong_missing_b_gradient(a,b):
    def back(g):a.grad+=g@b.data.T
    return Node(a.data@b.data,(a,b),back)

def wrong_reset_parents(a,b):
    def back(g):
        a.grad=np.zeros_like(a.data)
        b.grad=np.zeros_like(b.data)
        a.grad+=g@b.data.T
        b.grad+=a.data.T@g
    return Node(a.data@b.data,(a,b),back)

def wrong_missing_b_parent(a,b):
    def back(g):
        a.grad+=g@b.data.T
        b.grad+=a.data.T@g
    return Node(a.data@b.data,(a,),back)

negative_trials=[]
for name,function in [('overwrite',wrong_overwrite),('missing_b_gradient',wrong_missing_b_gradient),
                      ('reset_parents',wrong_reset_parents),('missing_b_parent',wrong_missing_b_parent)]:
    try:
        result=check_matmul(function)
        negative_trials.append({'name':name,'checker_passed':True,'output':result})
    except (AssertionError,ValueError) as error:
        negative_trials.append({'name':name,'checker_passed':False,'error_type':type(error).__name__,'error':str(error)})

# 두 별도 matmul이 같은 a를 쓰면 각 경로의 미분을 더해야 합니다.
counter_a=Node([[1.,2.]])
counter_b=Node([[3.],[4.]])
counter_c=Node([[5.],[6.]])
counter_loss=(wrong_reset_parents(counter_a,counter_b)+wrong_reset_parents(counter_a,counter_c)).sum()
counter_loss.backward()
wrong_shared_gradient=counter_a.grad.copy()
assert not np.allclose(wrong_shared_gradient,[[8.,10.]])
good_a=Node([[1.,2.]])
(my_matmul(good_a,Node([[3.],[4.]]))+my_matmul(good_a,Node([[5.],[6.]]))).sum().backward()
np.testing.assert_allclose(good_a.grad,[[8.,10.]])

# 한 부모를 목록에서 빼면 그 부모는 backward 초기화·방문 대상에서 빠집니다.
parent_a,parent_b=Node([[1.,2.]]),Node([[3.],[4.]])
parent_loss=wrong_missing_b_parent(parent_a,parent_b).sum()
parent_loss.backward()
parent_first=parent_b.grad.copy()
parent_loss.backward()
parent_second=parent_b.grad.copy()
np.testing.assert_allclose(parent_first,[[1.],[2.]])
np.testing.assert_allclose(parent_second,[[2.],[4.]])
metrics['intentional_errors']={'checker_trials':negative_trials,
    'reset_parent_counterexample':{'loss':float(counter_loss.data),'expected_shared_gradient':[[8.,10.]],
        'observed_shared_gradient':wrong_shared_gradient.tolist(),'own_correct_gradient':good_a.grad.tolist()},
    'missing_parent_counterexample':{'first_backward_b':parent_first.tolist(),'second_backward_b':parent_second.tolist(),
        'expected_after_each_backward':[[1.],[2.]]}}

Path('/tmp/dlfs-bridges/rehearsal13/metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2))
print(json.dumps(metrics,ensure_ascii=False,indent=2))
