import json

# 수식 Y[n,k]=sum_d A[n,d]*B[d,k]에서 순전파와 두 VJP를 별도 작성합니다.
# 기존 matmul을 호출하거나 별명으로 만들지 않습니다.
def my_matmul(a, b):
    if a.data.ndim != 2 or b.data.ndim != 2:
        raise ValueError('두 입력 Node의 값은 2차원 행렬이어야 합니다.')
    result = Node(a.data @ b.data, parents=(a, b))
    def propagate(upstream):
        left_gradient = upstream @ b.data.T
        right_gradient = a.data.T @ upstream
        a.grad += left_gradient
        b.grad += right_gradient
    result.pullback = propagate
    return result

my_checker_result = check_matmul(my_matmul)
print('직접 쓴 matmul:', my_checker_result)

# 작은 사각 행렬에서 곱과 두 부모 미분을 손계산합니다.
hand_A = np.array([[1., 2., 3.], [4., 5., 6.]])
hand_B = np.array([[2., -1.], [0., 3.], [1., 2.]])
hand_G = np.array([[1., -2.], [.5, 3.]])
hand_a, hand_b = Node(hand_A), Node(hand_B)
hand_output = my_matmul(hand_a, hand_b)
assert hand_output.parents == (hand_a, hand_b)
hand_output.backward(hand_G)
np.testing.assert_allclose(hand_output.data, [[5., 11.], [14., 23.]])
np.testing.assert_allclose(hand_a.grad, [[4., -6., -3.], [-2., 9., 6.5]])
np.testing.assert_allclose(hand_b.grad, [[3., 10.], [4.5, 11.], [6., 12.]])
hand_numeric_errors = [
    rel_error(hand_a.grad, numerical_gradient(lambda: np.sum(my_matmul(Node(hand_A), Node(hand_B)).data*hand_G), hand_A)),
    rel_error(hand_b.grad, numerical_gradient(lambda: np.sum(my_matmul(Node(hand_A), Node(hand_B)).data*hand_G), hand_B)),
]
assert max(hand_numeric_errors) < 1e-7

# 한 Node가 두 입력인 경우와 반복 호출, seed를 바꾸는 경우입니다.
shared = Node(np.array([[1., 2.], [3., 4.]]))
shared_G = np.array([[.5, -1.], [2., .3]])
shared_result = my_matmul(shared, shared)
shared_result.backward(shared_G)
shared_first = shared.grad.copy()
np.testing.assert_allclose(shared_first, [[5., -2.6], [11.6, 6.4]])
shared_result.backward(shared_G)
shared_repeat_difference = float(np.max(np.abs(shared.grad-shared_first)))
np.testing.assert_allclose(shared.grad, shared_first)
shared_result.backward(2*shared_G)
np.testing.assert_allclose(shared.grad, 2*shared_first)

# 공유 입력의 서로 다른 두 연산, 오른쪽 입력의 앞선 부모까지 실제로 따라갑니다.
branch_a = Node(hand_A)
branch_b, branch_c = Node(hand_B), Node(hand_B * .3)
branch_output = my_matmul(branch_a, branch_b) + my_matmul(branch_a, branch_c)
branch_output.backward(hand_G)
branch_expected = hand_G @ (branch_b.data+branch_c.data).T
branch_error = rel_error(branch_a.grad, branch_expected)
assert branch_error < 1e-12
right_root = Node(hand_B)
right_chain = my_matmul(Node(hand_A), right_root*2.)
right_chain.backward(hand_G)
right_chain_gradient = right_root.grad.copy()
np.testing.assert_allclose(right_chain_gradient, 2*hand_A.T @ hand_G)
left_root = Node(hand_A)
left_chain = my_matmul(left_root*3., Node(hand_B))
left_chain.backward(hand_G)
left_chain_gradient = left_root.grad.copy()
np.testing.assert_allclose(left_chain_gradient, 3*hand_G @ hand_B.T)

# 양쪽 broadcasting 및 스칼라 broadcasting을 비상수 seed로 비교합니다.
row2 = Node(np.array([[1.], [2.]]))
column2 = Node(np.array([3., 4., 5.]))
scalar2 = Node(.5)
broadcast_seed = np.arange(1., 7.).reshape(2, 3)
broadcast_value = row2*column2 + column2 + scalar2
broadcast_value.backward(broadcast_seed)
np.testing.assert_allclose(row2.grad, [[26.], [62.]])
np.testing.assert_allclose(column2.grad, [14., 19., 24.])
np.testing.assert_allclose(scalar2.grad, 21.)
tr = torch.tensor(row2.data, dtype=torch.float64, requires_grad=True)
tc = torch.tensor(column2.data, dtype=torch.float64, requires_grad=True)
ts = torch.tensor(scalar2.data, dtype=torch.float64, requires_grad=True)
(tr*tc+tc+ts).backward(torch.tensor(broadcast_seed, dtype=torch.float64))
broadcast_torch_errors = [rel_error(node.grad, t.grad.numpy()) for node,t in [(row2,tr),(column2,tc),(scalar2,ts)]]
assert max(broadcast_torch_errors) < 1e-12

# 같은 그래프의 반복 backward 규약: Node는 초기화, PyTorch는 누적합니다.
sx = Node(3.)
su = sx*sx
sy = su+su
sy.backward()
node_once = float(sx.grad)
sy.backward()
node_twice = float(sx.grad)
tx = torch.tensor(3., dtype=torch.float64, requires_grad=True)
tu = tx*tx
ty = tu+tu
ty.backward(retain_graph=True)
torch_once = tx.grad.item()
ty.backward(retain_graph=True)
torch_twice = tx.grad.item()
tx.grad = None
ty.backward()
torch_after_clear = tx.grad.item()
np.testing.assert_allclose([node_once,node_twice,torch_once,torch_twice,torch_after_clear],[12.,12.,12.,24.,12.])

# 의도적 오답: 공유 기울기 덮어쓰기, 한쪽 부모 누락, seed 무시.
def wrong_overwrite(a, b):
    def pull(g):
        a.grad = g @ b.data.T
        b.grad = a.data.T @ g
    return Node(a.data @ b.data, (a,b), pull)


def wrong_missing_right_parent(a, b):
    def pull(g):
        a.grad += g @ b.data.T
        b.grad += a.data.T @ g
    return Node(a.data @ b.data, (a,), pull)


def wrong_missing_left_parent(a, b):
    def pull(g):
        a.grad += g @ b.data.T
        b.grad += a.data.T @ g
    return Node(a.data @ b.data, (b,), pull)


def wrong_ignores_seed(a, b):
    def pull(g):
        ones = np.ones_like(g)
        a.grad += ones @ b.data.T
        b.grad += a.data.T @ ones
    return Node(a.data @ b.data, (a,b), pull)

negative_results = {}
for name, candidate in [('overwrite_shared_gradient',wrong_overwrite),
                         ('missing_right_parent',wrong_missing_right_parent),
                         ('missing_left_parent',wrong_missing_left_parent),
                         ('ignores_upstream_seed',wrong_ignores_seed)]:
    try:
        outcome = check_matmul(candidate)
    except AssertionError as exc:
        negative_results[name] = {'rejected':True,'error_type':type(exc).__name__,'message':str(exc).strip()}
    else:
        negative_results[name] = {'rejected':False,'outcome':outcome}
assert all(item['rejected'] for item in negative_results.values())

# 부모를 빼면 leaf 값에는 기울기가 생겨도 앞선 곱셈의 root에는 전달되지 않습니다.
wrong_root = Node(hand_B)
wrong_missing_right_parent(Node(hand_A), wrong_root*2.).backward(hand_G)
missing_parent_counterexample = {'actual_root_grad':wrong_root.grad.tolist(),
                                 'expected_root_grad':(2*hand_A.T @ hand_G).tolist()}
assert np.all(wrong_root.grad==0)
assert np.any(2*hand_A.T @ hand_G != 0)

# 새로운 N=3,D=2,H=6,K=2 MLP에 독립 matmul을 넣고 4가지 미분을 대조합니다.
check_rng = np.random.default_rng(1307)
my_X = check_rng.normal(size=(3,2))
my_Y = check_rng.normal(size=(3,2))
my_values = [check_rng.normal(size=s)*.25 for s in [(2,6),(6,),(6,2),(2,)]]
my_parameters = [Node(v) for v in my_values]
my_loss = graph_loss(my_X, my_Y, my_parameters, matmul_fn=my_matmul)
my_loss.backward()
my_manual_value, my_manual_grads = manual_mlp(my_X, my_Y, my_values)
np.testing.assert_allclose(my_loss.data, my_manual_value)
my_engine_errors = [rel_error(p.grad, g) for p,g in zip(my_parameters, my_manual_grads)]
my_numeric_errors = [rel_error(p.grad, numerical_gradient(lambda: manual_mlp(my_X,my_Y,my_values)[0], arr))
                     for p,arr in zip(my_parameters,my_values)]
my_torch_params = [torch.tensor(v,dtype=torch.float64,requires_grad=True) for v in my_values]
my_tX = torch.tensor(my_X,dtype=torch.float64)
my_tY = torch.tensor(my_Y,dtype=torch.float64)
w1, b1, w2, b2 = my_torch_params
my_tloss = .5*((torch.tanh(my_tX @ w1+b1) @ w2+b2 - my_tY)**2).mean()
my_tloss.backward()
my_torch_errors = [rel_error(p.grad,t.grad.numpy()) for p,t in zip(my_parameters,my_torch_params)]
assert max(my_engine_errors) < 1e-12 and max(my_numeric_errors)<1e-7 and max(my_torch_errors)<1e-12
np.testing.assert_allclose(my_tloss.item(), my_loss.data, atol=1e-14)

# 본문의 직접 변형 지시를 그대로 수행합니다: 후보 전달 + 너비 6 + 실제 1800스텝.
my_trained_parameters, my_train_history = fit_graph_mlp(
    X_train,Y_train,hidden_size=6,steps=1800,matmul_fn=my_matmul)
my_valid_prediction = graph_prediction(X_valid,my_trained_parameters,matmul_fn=my_matmul).data
my_valid_half_mse = .5*np.mean((my_valid_prediction-Y_valid)**2)
assert my_train_history[-1] < .05*my_train_history[0]
assert my_valid_half_mse < .02
np.testing.assert_allclose(my_train_history,smaller_history,atol=1e-14)
my_grid_prediction = graph_prediction(grid,my_trained_parameters,matmul_fn=my_matmul).data
plot_curves(np.arange(len(my_train_history)),
            {'기본 너비 12':train_history,'직접 matmul + 너비 6':my_train_history},
            title='독립 연산으로 실제 너비 변경 학습',ylabel='학습 half-MSE',logy=True)
plot_clouds([('직접 matmul·너비 6의 예측',{
    '학습 관측':np.column_stack([X_train,Y_train]),
    '학습 함수':np.column_stack([grid,my_grid_prediction]),
    '잡음 전 함수':np.column_stack([grid,np.sin(np.pi*grid)])})],
    xlabel='입력 x',ylabel='출력 y',equal=False)

rehearsal_metrics={
    'checker_result':my_checker_result,
    'hand_matmul':{'output':hand_output.data.tolist(),'da':hand_a.grad.tolist(),'db':hand_b.grad.tolist(),
                   'gradient_relative_errors':hand_numeric_errors},
    'shared_paths':{'double_parent_gradient':shared_first.tolist(),'repeat_max_abs_difference':shared_repeat_difference,
                    'branch_relative_error':branch_error,'right_chain_gradient':right_chain_gradient.tolist(),
                    'left_chain_gradient':left_chain_gradient.tolist()},
    'broadcast':{'row_grad':row2.grad.tolist(),'column_grad':column2.grad.tolist(),'scalar_grad':float(scalar2.grad),
                  'pytorch_gradient_errors':broadcast_torch_errors},
    'repeated_backward':{'node_once':node_once,'node_twice':node_twice,'torch_once':torch_once,
                         'torch_twice':torch_twice,'torch_after_grad_none':torch_after_clear},
    'negative_candidates':negative_results,'missing_parent_counterexample':missing_parent_counterexample,
    'new_mlp':{'N':3,'D':2,'H':6,'K':2,'loss':float(my_loss.data),
               'manual_gradient_relative_errors':my_engine_errors,'numeric_gradient_relative_errors':my_numeric_errors,
               'pytorch_gradient_relative_errors':my_torch_errors},
    'width_variant':{'hidden_size':6,'steps':1800,'parameter_count':sum(p.data.size for p in my_trained_parameters),
                     'initial_train_half_mse':float(my_train_history[0]),'final_train_half_mse':float(my_train_history[-1]),
                     'valid_half_mse':float(my_valid_half_mse),'baseline_history_max_abs_difference':float(np.max(np.abs(my_train_history-smaller_history)))},
    'original_notebook':{'manual_errors':engine_errors,'numeric_errors':numeric_errors,'pytorch_errors':torch_errors,
                          'baseline_width':12,'baseline_parameter_count':sum(p.data.size for p in trained_parameters),
                          'initial_train_half_mse':float(train_history[0]),'final_train_half_mse':float(train_history[-1]),
                          'valid_half_mse':float(valid_half_mse),
                          'width6_train_half_mse':float(smaller_history[-1]),'width6_valid_half_mse':float(smaller_valid),
                          'stop_gradient_full':float(full_gradient),'stop_gradient_stopped':float(stopped_gradient)},
}
print('REHEARSAL_METRICS='+json.dumps(rehearsal_metrics,ensure_ascii=False))
