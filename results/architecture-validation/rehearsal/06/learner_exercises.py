import contextlib
import io
import json
from pathlib import Path

metrics = {}

# 시간별 전체 누적 미분을 배열로 보관하는 방식으로 직접 작성합니다.
# forward의 states[:,0]이 h0이므로 미분 배열도 같은 인덱스를 사용합니다.
def my_rnn_backward(dH, cache, p):
    X, states = cache
    N, T, D = X.shape
    width = states.shape[-1]
    accumulated = np.zeros_like(states)
    accumulated[:, 1:, :] = dH
    preactivation_gradient = np.zeros_like(dH)
    for time in range(T, 0, -1):
        preactivation_gradient[:, time-1] = accumulated[:, time] * (1-states[:, time]**2)
        accumulated[:, time-1] += preactivation_gradient[:, time-1] @ p['Wh'].T
    # 각 (표본,시각)을 같은 순서로 펼쳐, 공유 파라미터의 모든 기여를 합합니다.
    input_rows = X.reshape(N*T, D)
    state_rows = states[:, :-1].reshape(N*T, width)
    gradient_rows = preactivation_gradient.reshape(N*T, width)
    parameter_gradients = {
        'Wx': input_rows.T @ gradient_rows,
        'Wh': state_rows.T @ gradient_rows,
        'b': gradient_rows.sum(axis=0),
    }
    return preactivation_gradient @ p['Wx'].T, accumulated[:, 0], parameter_gradients


own_errors = check_rnn_backward(my_rnn_backward)
metrics['own_backward'] = {'errors': own_errors, 'max_relative_error': max(own_errors.values())}
print('직접 작성한 RNN backward 검사:', metrics['own_backward'])

# 본문의 가장 작은 forward를 그대로 사용해, L=h2의 두 스텝 backward를 손으로 펼칩니다.
tiny_H, tiny_cache = rnn_forward(hand_X, hand_p)
tiny_G = np.zeros_like(tiny_H)
tiny_G[:, -1] = 1
tiny_dx, tiny_dh0, tiny_dp = my_rnn_backward(tiny_G, tiny_cache, hand_p)
delta2 = 1-h2**2
delta1 = 0.5 * delta2 * (1-h1**2)
np.testing.assert_allclose(tiny_dx, [[[delta1], [delta2]]])
np.testing.assert_allclose(tiny_dh0, [[0.5*delta1]])
np.testing.assert_allclose(tiny_dp['Wx'], [[delta1]])
np.testing.assert_allclose(tiny_dp['Wh'], [[h1*delta2]])
np.testing.assert_allclose(tiny_dp['b'], [delta1+delta2])
metrics['two_step_hand_derivation'] = {
    'loss': 'L=h2', 'h1': float(h1), 'h2': float(h2),
    'delta1':float(delta1), 'delta2':float(delta2),
    'dX': tiny_dx.tolist(), 'dh0':tiny_dh0.tolist(),
    'grads':{k:v.tolist() for k,v in tiny_dp.items()},
    'dX_shape':list(tiny_dx.shape), 'dh0_shape':list(tiny_dh0.shape),
}

# 모든 시각의 손실 L=0.5*sum(H**2)에서 입력·초기상태·모든 공유 파라미터를 검사합니다.
all_rng = np.random.default_rng(906)
all_X = all_rng.normal(size=(2, 4, 2))*0.2
all_h0 = all_rng.normal(size=(2,3))*0.2
all_p = {'Wx':all_rng.normal(size=(2,3))*0.2,
         'Wh':all_rng.normal(size=(3,3))*0.2, 'b':all_rng.normal(size=3)*0.1}
all_H, all_cache = rnn_forward(all_X, all_p, all_h0)
all_dx, all_dh0, all_dp = my_rnn_backward(all_H, all_cache, all_p)
all_loss = lambda: float(0.5*np.sum(rnn_forward(all_X, all_p, all_h0)[0]**2))
all_errors = {}
all_shapes = {}
for name, variable, analytic in [('X',all_X,all_dx),('h0',all_h0,all_dh0),
                                *[(key,value,all_dp[key]) for key,value in all_p.items()]]:
    numeric_value = numerical_gradient(all_loss, variable)
    error = rel_error(analytic, numeric_value)
    assert error < 2e-6, (name, error)
    all_errors[name] = error
    all_shapes[name] = list(analytic.shape)
metrics['all_time_loss'] = {'loss':all_loss(), 'objective':'0.5 * sum(H**2)',
    'direct_gradient':'dH=H', 'H_shape':list(all_H.shape), 'gradient_shapes':all_shapes,
    'errors':all_errors, 'max_relative_error':max(all_errors.values())}

# 의도적 오답 1: 미래 누적을 제거하지만 T=1에서의 dh0는 올바르게 반환합니다.
def wrong_future_only(dH, cache, p):
    X, states = cache
    N,T,D = X.shape
    da = dH * (1-states[:,1:]**2)
    rows = da.reshape(N*T, -1)
    grads = {'Wx': X.reshape(N*T,D).T @ rows,
             'Wh': states[:,:-1].reshape(N*T,-1).T @ rows,
             'b': rows.sum(axis=0)}
    return da @ p['Wx'].T, da[:,0] @ p['Wh'].T, grads


# 의도적 오답 2: 시간 경로는 맞지만 공유 파라미터는 첫 시각의 기여만 남깁니다.
def wrong_shared_accumulation(dH, cache, p):
    X, states = cache
    N,T,_ = X.shape
    da_all = np.zeros_like(dH)
    total = np.zeros_like(states)
    total[:,1:] = dH
    for time in range(T,0,-1):
        da_all[:,time-1] = total[:,time]*(1-states[:,time]**2)
        total[:,time-1] += da_all[:,time-1] @ p['Wh'].T
    da = da_all[:,0]
    wrong_grads = {'Wx':X[:,0].T@da, 'Wh':states[:,0].T@da, 'b':da.sum(axis=0)}
    return da_all@p['Wx'].T, total[:,0], wrong_grads


negative_results = []
for name, function in [('notebook_wrong_no_future', wrong_no_future),
                       ('future_path_only', wrong_future_only),
                       ('shared_parameter_sum', wrong_shared_accumulation)]:
    try:
        check_rnn_backward(function)
    except AssertionError as error:
        negative_results.append({'name':name, 'detected':True, 'error':str(error)})
    else:
        negative_results.append({'name':name, 'detected':False})
assert all(result['detected'] for result in negative_results)
assert negative_results[1]['error'].startswith('(4, ')
assert negative_results[2]['error'].startswith('(4, ')
metrics['intentional_errors'] = negative_results

# 길이 6 본문 학습 결과를 보존합니다. 새 파라미터·optimizer 상태로 길이 4를 학습합니다.
def parameter_count(recurrence, classifier):
    rnn_count = sum(value.size for value in recurrence.values())
    head_count = sum(value.size for value in classifier.values())
    return {'rnn':rnn_count, 'head':head_count, 'total':rnn_count+head_count}

metrics['length6_baseline'] = {
    'train_count':int(len(y_train)), 'valid_count':int(len(y_valid)),
    'train_length':int(X_train.shape[1]), 'valid_length':int(X_valid.shape[1]),
    'train_ce_initial':float(history['train'][0]), 'train_ce_final':float(history['train'][-1]),
    'valid_ce_final':float(history['valid'][-1]), 'train_accuracy':float(train_accuracy),
    'valid_accuracy':float(valid_accuracy), 'parameters':parameter_count(p,head),
}
X4, y4, tokens4 = order_data(4)
p4, head4 = init_model()
original_p4 = {key:value.copy() for key,value in p4.items()}
state4 = {key:(np.zeros_like(value),np.zeros_like(value)) for key,value in p4.items()}
statehead4 = {key:(np.zeros_like(value),np.zeros_like(value)) for key,value in head4.items()}
train4_log = []
for step in range(401):
    loss4, logits4, grads4, dhead4, _ = model_loss(X4,y4,p4,head4,backward_fn=my_rnn_backward)
    if step%40==0:
        validation_loss4, validation_logits4, *_ = model_loss(X_valid,y_valid,p4,head4,backward_fn=my_rnn_backward)
        train4_log.append({'step':step,'train_ce':float(loss4),'valid_ce':float(validation_loss4)})
    if step==400:
        break
    norm4 = np.sqrt(sum(np.sum(g*g) for g in [*grads4.values(),*dhead4.values()]))
    factor4 = min(1.,1./(norm4+1e-12))
    clipped4 = {key:value*factor4 for key,value in grads4.items()}
    clippedhead4 = {key:value*factor4 for key,value in dhead4.items()}
    adam_step(p4,clipped4,state4,step+1)
    adam_step(head4,clippedhead4,statehead4,step+1)
accuracy4 = np.mean(logits4.argmax(axis=1)==y4)
valid_accuracy4 = np.mean(validation_logits4.argmax(axis=1)==y_valid)
assert accuracy4==1. and loss4<0.05
assert parameter_count(p4,head4)==metrics['length6_baseline']['parameters']
assert parameter_count(p4,head4)['rnn']==3*12+12*12+12
metrics['length4_variation'] = {
    'train_shape':list(X4.shape), 'valid_shape':list(X_valid.shape), 'updates':400,
    'fresh_model_and_optimizer_state':True, 'backward_used':'my_rnn_backward',
    'train_accuracy':float(accuracy4), 'valid_accuracy':float(valid_accuracy4),
    'train_ce_final':float(loss4), 'valid_ce_final':float(validation_loss4),
    'parameters':parameter_count(p4,head4), 'history':train4_log,
}

# 도함수 시각화의 입력 기울기는 초기상태 기울기와 다른 지수를 갖습니다.
# t=1..T일 때 zero-state scalar에서는 dh_T/dx_t=Wh**(T-t), dh_T/dh0=Wh**T입니다.
temporal_checks = {}
for weight in [0.7,1.,1.2]:
    values = temporal_gradients[f'Wh={weight}']
    expected_values = weight**(25-np.arange(1,26))
    np.testing.assert_allclose(values, expected_values, rtol=1e-12)
    temporal_checks[str(weight)] = {'first_input_gradient':float(values[0]),
                                  'last_input_gradient':float(values[-1]),
                                  'initial_state_gradient':float(weight**25),
                                  'max_absolute_error':float(np.max(np.abs(values-expected_values)))}
metrics['temporal_gradient_formulas'] = temporal_checks

Path('/tmp/dlfs-architectures/rehearsal06/metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2))
print(json.dumps(metrics,ensure_ascii=False,indent=2))
