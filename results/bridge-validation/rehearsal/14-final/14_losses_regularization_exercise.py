import json

# 세 잔차 손실과 가중 평균을 관측·출력 반복문으로 독립 작성합니다.
# element_loss 또는 원본 regression_loss는 호출하지 않습니다.
def my_regression_loss(prediction, target, kind='huber', delta=.5, weights=None):
    if prediction.ndim != 2 or target.shape != prediction.shape:
        raise ValueError('예측·정답의 shape는 같은 (N,K)입니다.')
    N,K = prediction.shape
    a = np.ones(N) if weights is None else np.asarray(weights,dtype=float)
    if a.shape!=(N,) or np.any(a<0) or not np.all(np.isfinite(a)) or a.sum()<=0:
        raise ValueError('관측 가중치는 비음수이며 합이 양수입니다.')
    if kind not in ['mse','mae','huber'] or (kind=='huber' and delta<=0):
        raise ValueError('세 손실 이름 또는 양의 Huber delta가 필요합니다.')
    total = 0.
    d_prediction = np.zeros_like(prediction,dtype=float)
    denominator = K * a.sum()
    for n in range(N):
        for k in range(K):
            r = prediction[n,k]-target[n,k]
            if kind=='mse':
                value,derivative = r*r/2, r
            elif kind=='mae':
                value = abs(r)
                derivative = 1. if r>0 else (-1. if r<0 else 0.)
            elif abs(r)<=delta:
                value,derivative = r*r/2, r
            else:
                value = delta*abs(r)-delta*delta/2
                derivative = delta if r>0 else -delta
            total += a[n]*value/denominator
            d_prediction[n,k] = a[n]*derivative/denominator
    return total,d_prediction

checker_result = check_regression_loss(my_regression_loss)
print('직접 쓴 회귀 손실:',checker_result)

# N=2,K=2 손계산: (0+.875 + 3*(.875+.03125)) / (2*4)
hand_pred = np.array([[0.,2.],[-2.,.25]])
hand_weights = np.array([1.,3.])
hand_loss,hand_grad = my_regression_loss(hand_pred,np.zeros_like(hand_pred),'huber',.5,hand_weights)
np.testing.assert_allclose(hand_loss,.44921875)
np.testing.assert_allclose(hand_grad,[[0.,.0625],[-.1875,.09375]])

# 새 N=4,K=3: 가중치 0, 전체 가중치 배율, 반복 관측을 확인합니다.
my_rng = np.random.default_rng(1407)
new_pred = my_rng.normal(size=(4,3))
new_target = my_rng.normal(size=(4,3))
new_weights = np.array([.5,0.,3.,1.])
new_errors = {}
for kind in ['mse','mae','huber']:
    value,grad = my_regression_loss(new_pred,new_target,kind,.47,new_weights)
    numeric = numerical_gradient(lambda: my_regression_loss(new_pred,new_target,kind,.47,new_weights)[0],new_pred)
    new_errors[kind] = rel_error(grad,numeric)
    assert new_errors[kind]<1e-7
    np.testing.assert_allclose(grad[1],0.)
    scaled_value,scaled_grad = my_regression_loss(new_pred,new_target,kind,.47,7*new_weights)
    np.testing.assert_allclose([scaled_value], [value])
    np.testing.assert_allclose(scaled_grad,grad)
    rv,rg = my_regression_loss(np.repeat(new_pred,2,axis=0),np.repeat(new_target,2,axis=0),kind,.47,np.repeat(new_weights,2))
    np.testing.assert_allclose(rv,value)
    np.testing.assert_allclose(rg,np.repeat(grad,2,axis=0)/2)

# 의도적 오답: 출력 평균 누락, 관측 가중치 무시, Huber 상수 누락, 기울기 clipping 누락.
def wrong_missing_output_mean(pred,target,kind,delta,weights):
    value,grad = my_regression_loss(pred,target,kind,delta,weights)
    return value*pred.shape[1],grad*pred.shape[1]


def wrong_ignores_weights(pred,target,kind,delta,weights):
    return my_regression_loss(pred,target,kind,delta,None)


def wrong_huber_linear_constant(pred,target,kind,delta,weights):
    value,grad = my_regression_loss(pred,target,kind,delta,weights)
    if kind=='huber':
        a = np.ones(len(pred)) if weights is None else np.asarray(weights)
        linear = np.abs(pred-target)>delta
        value += .5*delta**2*np.sum(linear*a[:,None])/(a.sum()*pred.shape[1])
    return value,grad


def wrong_unclipped_huber_gradient(pred,target,kind,delta,weights):
    value,grad = my_regression_loss(pred,target,kind,delta,weights)
    if kind=='huber':
        a = np.ones(len(pred)) if weights is None else np.asarray(weights)
        grad = (pred-target)*a[:,None]/(a.sum()*pred.shape[1])
    return value,grad

negative_results = {}
for name,candidate in [('missing_output_mean',wrong_missing_output_mean),('ignores_weights',wrong_ignores_weights),
                       ('missing_huber_constant',wrong_huber_linear_constant),('unclipped_huber_gradient',wrong_unclipped_huber_gradient)]:
    try:
        result = check_regression_loss(candidate)
    except AssertionError as exc:
        negative_results[name] = {'rejected':True,'error_type':type(exc).__name__,'message':str(exc).strip()}
    else:
        negative_results[name] = {'rejected':False,'outcome':result}
assert all(item['rejected'] for item in negative_results.values())

# 독립 손실 함수를 실제 fit_linear에 전달합니다. delta 세 값을 각 1200스텝 학습합니다.
my_delta_results = {}
my_prediction_curves = {'깨끗한 함수':Y_clean[:,0]}
for my_delta in [.1,.3,1.]:
    my_W,my_b,my_history = fit_linear(X_train,Y_train,'huber',delta=my_delta,steps=1200,loss_fn=my_regression_loss)
    my_prediction = X_clean @ my_W+my_b
    my_rmse = np.sqrt(np.mean((my_prediction-Y_clean)**2))
    np.testing.assert_allclose(my_rmse,delta_results[my_delta],atol=1e-13)
    assert my_history[-1]<my_history[0]
    my_delta_results[str(my_delta)]={'W':my_W.tolist(),'b':my_b.tolist(),'initial_objective':float(my_history[0]),
                                    'final_objective':float(my_history[-1]),'clean_rmse':float(my_rmse),'steps':1200}
    my_prediction_curves[f'직접 손실 delta={my_delta}']=my_prediction[:,0]
plot_curves(X_clean[:,0],my_prediction_curves,title='독립 Huber 손실의 delta 변형 학습',xlabel='x',ylabel='회귀 예측')
plot_bars(list(my_delta_results),[v['clean_rmse'] for v in my_delta_results.values()],
          title='같은 깨끗한 함수에서 delta 비교',ylabel='RMSE')

# 독립 손실로 K=3 ridge를 학습하여 NK*lambda 계수를 검산합니다.
my_Xr = my_rng.normal(size=(40,2))
my_Yr = my_Xr @ np.array([[1.,-.5,1.5],[-2.,.7,.3]])+np.array([.4,-.2,.6])
my_Yr += my_rng.normal(size=my_Yr.shape)*.04
my_Wr,my_br,my_hr = fit_linear(my_Xr,my_Yr,'mse',l2=.4,steps=2500,loss_fn=my_regression_loss)
my_A = np.column_stack([my_Xr,np.ones(len(my_Xr))])
my_P = np.diag([1.,1.,0.])
my_exact = np.linalg.solve(my_A.T @ my_A+len(my_A)*my_Yr.shape[1]*.4*my_P,my_A.T @ my_Yr)
my_theta = np.vstack([my_Wr,my_br])
my_ridge_error = float(np.max(np.abs(my_theta-my_exact)))
assert my_ridge_error<1e-10
wrong_exact = np.linalg.solve(my_A.T @ my_A+len(my_A)*.4*my_P,my_A.T @ my_Yr)
wrong_ridge_difference = float(np.max(np.abs(my_exact-wrong_exact)))
assert wrong_ridge_difference>.01

# AdamW 두 스텝을 수식으로 별도 전개합니다. epsilon은 제곱근 바깥입니다.
aw = np.array([1.2,-.7]); am=np.zeros(2); av=np.zeros(2); astate=(np.zeros(2),np.zeros(2),0)
adamw_steps=[]
for t,ag in enumerate([np.array([2.,1e-9]),np.array([-1.,2e-9])],1):
    am = .5*am+.5*ag
    av = .75*av+.25*ag**2
    mhat=am/(1-.5**t); vhat=av/(1-.75**t)
    expected_aw = .98*aw-.1*mhat/(np.sqrt(vhat)+.03)
    actual_aw,astate = adam_step(aw,ag,astate,lr=.1,beta1=.5,beta2=.75,eps=.03,weight_decay=.2)
    np.testing.assert_allclose(actual_aw,expected_aw,atol=1e-14)
    np.testing.assert_allclose(astate[0],am,atol=1e-15)
    np.testing.assert_allclose(astate[1],av,atol=1e-30)
    epsilon_inside = .98*aw-.1*mhat/np.sqrt(vhat+.03)
    adamw_steps.append({'t':t,'m':am.tolist(),'v':av.tolist(),'mhat':mhat.tolist(),'vhat':vhat.tolist(),
                        'W':actual_aw.tolist(),'epsilon_inside_W':epsilon_inside.tolist(),
                        'epsilon_position_difference':float(np.max(np.abs(actual_aw-epsilon_inside)))})
    aw=actual_aw

# 고정 mask의 손계산과 dropout의 기대 손실을 별도 q=.5에서 확인합니다.
my_H = np.array([[1.,-2.],[.5,3.]])
my_G = np.array([[.3,-2.],[1.5,.7]])
my_mask = np.array([[1.,0.],[0.,1.]])
my_q=.5
my_dropout = my_H*my_mask/my_q
my_dH = my_G*my_mask/my_q
np.testing.assert_allclose(my_dropout,[[2.,0.],[0.,6.]])
np.testing.assert_allclose(my_dH,[[.6,0.],[0.,1.4]])
my_dropout_error=rel_error(my_dH,numerical_gradient(lambda:np.sum(my_H*my_mask/my_q*my_G),my_H))
assert my_dropout_error<1e-7
my_expected_loss=0.
for n in range(len(X_two)):
    for mask in masks:
        p_mask=np.prod(np.where(mask==1,my_q,1-my_q))
        output=np.sum(X_two[n]*mask/my_q*w_two)+b_two
        my_expected_loss+=p_mask*.5*(output-y_two[n])**2/len(X_two)
my_variance_term=(1-my_q)/(2*my_q)*np.sum(w_two**2*np.mean(X_two**2,axis=0))
np.testing.assert_allclose(my_expected_loss,base_loss+my_variance_term,atol=1e-14)
# f(h)=h^2의 비선형 반례: H=2, q=.5 -> E[f(drop(H))]=8, f(E[drop(H)])=4.
nonlinear_expected=.5*0**2+.5*4**2
assert nonlinear_expected==8 and nonlinear_expected!=2**2

rehearsal_metrics={
 'checker_result':checker_result,'hand_weighted_huber':{'loss':float(hand_loss),'gradient':hand_grad.tolist()},
 'new_weighted_loss':{'N':4,'K':3,'gradient_relative_errors':new_errors,'zero_weight_row_gradient_is_zero':True,
                      'weight_scaling_invariance':True,'repeated_batch_invariance':True},
 'negative_candidates':negative_results,'delta_training':my_delta_results,
 'new_ridge':{'N':40,'K':3,'l2':.4,'steps':2500,'closed_form_max_abs_difference':my_ridge_error,
               'missing_K_closed_form_difference':wrong_ridge_difference,'weight':my_Wr.tolist(),'bias':my_br.tolist()},
 'adamw_hand_steps':adamw_steps,
 'independent_dropout':{'q':my_q,'fixed_mask_output':my_dropout.tolist(),'fixed_mask_gradient':my_dH.tolist(),
                         'gradient_relative_error':float(my_dropout_error),'expected_loss':float(my_expected_loss),
                         'base_loss':float(base_loss),'variance_term':float(my_variance_term),
                         'nonlinear_expected_square':nonlinear_expected,'square_of_mean':4.},
 'original_notebook':{'loss_gradient_errors':loss_gradient_errors,'linear_gradient_errors':linear_gradient_errors,
                       'clean_rmse_by_loss':{k:float(v) for k,v in rmse_by_loss.items()},'ridge_norms':[float(v) for v in ridge_norms],
                       'ridge_K2_max_abs_difference':float(np.max(np.abs(np.vstack([W_multi,b_multi])-exact_multi))),
                       'adam_coupled_W':W_coupled.tolist(),'adamw_decoupled_W':W_decoupled.tolist(),
                       'coupled_m':coupled_state[0].tolist(),'decoupled_m':decoupled_state[0].tolist(),
                       'dropout_gradient_error':float(dropout_error),'dropout_mean_output_max_abs_difference':float(np.max(np.abs(monte_carlo_output-H))),
                       'dropout_exact_expected_loss':float(exact_expected_loss),'dropout_base_loss':float(base_loss),'dropout_extra_loss':float(extra_loss)},
}
print('REHEARSAL_METRICS='+json.dumps(rehearsal_metrics,ensure_ascii=False))
