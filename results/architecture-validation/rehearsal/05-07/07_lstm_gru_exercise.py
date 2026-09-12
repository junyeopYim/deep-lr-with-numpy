import json,time

# 네 게이트의 affine 미분을 따로 계산하고 연결합니다.
# c로 오는 두 경로를 더한 뒤 f,i,g로 분기하며 o는 h에서 직접 받습니다.
def my_lstm_step_backward(dh,dc_future,cache,p):
    joined,previous_c,forget,input_gate,output_gate,candidate,current_c,D=cache
    hidden_width=previous_c.shape[1]
    tanh_c=np.tanh(current_c)
    dmemory=dc_future+dh*output_gate*(1-tanh_c*tanh_c)
    d_forget_pre=(dmemory*previous_c)*forget*(1-forget)
    d_input_pre=(dmemory*candidate)*input_gate*(1-input_gate)
    d_output_pre=(dh*tanh_c)*output_gate*(1-output_gate)
    d_candidate_pre=(dmemory*input_gate)*(1-candidate*candidate)
    weight_blocks=[p['W'][:,j*hidden_width:(j+1)*hidden_width] for j in range(4)]
    gate_partials=[d_forget_pre,d_input_pre,d_output_pre,d_candidate_pre]
    djoined=np.zeros_like(joined)
    weight_derivatives=[]
    bias_derivatives=[]
    for gate_partial,weight_block in zip(gate_partials,weight_blocks):
        djoined+=gate_partial@weight_block.T
        weight_derivatives.append(joined.T@gate_partial)
        bias_derivatives.append(gate_partial.sum(axis=0))
    parameter_derivatives={'W':np.concatenate(weight_derivatives,axis=1),'b':np.concatenate(bias_derivatives)}
    return djoined[:,:D],djoined[:,D:],dmemory*forget,parameter_derivatives

independent_step_errors=check_lstm_step(my_lstm_step_backward)
independent_sequence_errors=check_gated_sequence('LSTM',my_lstm_step_backward)

# c의 상류를 지운 경우는 같은 후보의 두 검사 모두 거부해야 합니다.
def my_wrong_drop_c(dh,dc_future,cache,p):
    return my_lstm_step_backward(dh,np.zeros_like(dc_future),cache,p)
wrong_messages={}
for name,check in [('one_step',lambda:check_lstm_step(my_wrong_drop_c)),('three_steps',lambda:check_gated_sequence('LSTM',my_wrong_drop_c))]:
    try:
        check()
    except AssertionError as exc:
        wrong_messages[name]=str(exc)
    assert name in wrong_messages,name+' 검사에서 c 상류 누락을 검출하지 못했습니다.'

# 학습 전 파라미터와 실제 입력에 대한 초기 게이트를 먼저 기록합니다.
initial_bias1_p=init_gates('LSTM',2,8,78)
_,_,initial_bias1_cache=gated_forward(X_valid[:1],initial_bias1_p,'LSTM')
initial_bias1_forget=np.stack([item[2][0] for item in initial_bias1_cache])
baseline_lstm_p,baseline_lstm_head,baseline_lstm_history,baseline_lstm_accuracy=trained['LSTM']
_,_,learned_bias1_cache=gated_forward(X_valid[:1],baseline_lstm_p,'LSTM')
learned_bias1_forget=np.stack([item[2][0] for item in learned_bias1_cache])

# 7절 지시대로, 임시 노트북에서 init_gates의 forget bias 한 줄을 1에서 0으로 바꿉니다.
# train_memory의 나머지 데이터·seed·폭·350회 업데이트는 그대로 호출합니다.
def init_gates(kind,D,H,seed):
    rng=np.random.default_rng(seed)
    if kind=='LSTM':
        W=rng.normal(size=(D+H,4*H))/np.sqrt(D+H)
        b=np.zeros(4*H)
        b[:H]=0.
        return {'W':W,'b':b}
    return {key:value for gate in ['z','r','g']
            for key,value in [('W'+gate,rng.normal(size=(D+H,H))/np.sqrt(D+H)),('b'+gate,np.zeros(H))]}

initial_bias0_p=init_gates('LSTM',2,8,78)
np.testing.assert_array_equal(initial_bias0_p['W'],initial_bias1_p['W'])
np.testing.assert_array_equal(initial_bias0_p['b'][:8],np.zeros(8))
np.testing.assert_array_equal(initial_bias0_p['b'][8:],initial_bias1_p['b'][8:])
_,_,initial_bias0_cache=gated_forward(X_valid[:1],initial_bias0_p,'LSTM')
initial_bias0_forget=np.stack([item[2][0] for item in initial_bias0_cache])
zero_variation_started=time.perf_counter()
zero_p,zero_head,zero_history,zero_accuracy=train_memory('LSTM',X_train,y_train,X_valid,y_valid)
zero_variation_seconds=time.perf_counter()-zero_variation_started
_,_,learned_bias0_cache=gated_forward(X_valid[:1],zero_p,'LSTM')
learned_bias0_forget=np.stack([item[2][0] for item in learned_bias0_cache])
assert np.isfinite(zero_history['train'][-1]) and np.isfinite(zero_history['valid'][-1])
plot_curves(zero_history['step'],{'forget bias 1 검증':baseline_lstm_history['valid'],'forget bias 0 검증':zero_history['valid']},title='직접 변형: forget bias만 바꾸고 같은 길이 12에서 학습',ylabel='BCE')
plot_matrices([learned_bias1_forget.T,learned_bias0_forget.T],['bias 1로 시작: 학습 후 forget gate','bias 0으로 시작: 학습 후 forget gate'],annotate=False)

def gate_summary(values):
    return {'shape':list(values.shape),'mean':float(values.mean()),'min':float(values.min()),'max':float(values.max()),'values':values.tolist()}

long_data,long_y=memory_data(64,24,703)
baseline_metrics={}
for kind,(params,head,hist,acc) in trained.items():
    long_loss,long_logits,*_=memory_loss(long_data,long_y,params,head,kind)
    baseline_metrics[kind]={'parameters':sum(v.size for v in [*params.values(),*head.values()]),'train_bce_initial':float(hist['train'][0]),'train_bce_final':float(hist['train'][-1]),'validation_bce':float(hist['valid'][-1]),'validation_accuracy':float(acc),'length24_bce':float(long_loss),'length24_accuracy':float(np.mean((long_logits>0)==long_y))}
zero_long_loss,zero_long_logits,*_=memory_loss(long_data,long_y,zero_p,zero_head,'LSTM')
metrics={'independent_lstm_step_errors':{k:float(v) for k,v in independent_step_errors.items()},'independent_lstm_sequence_errors':{k:float(v) for k,v in independent_sequence_errors.items()},'dropped_c_rejections':wrong_messages,'baseline':baseline_metrics,'forget_bias_zero_variation':{'steps':350,'width':8,'seed':78,'initial_bias_only_sigmoid':{'bias1':float(sigmoid(np.array([1.]))[0]),'bias0':float(sigmoid(np.array([0.]))[0])},'train_bce_initial':float(zero_history['train'][0]),'train_bce_final':float(zero_history['train'][-1]),'validation_bce':float(zero_history['valid'][-1]),'validation_accuracy':float(zero_accuracy),'training_seconds':zero_variation_seconds,'length24_bce':float(zero_long_loss),'length24_accuracy':float(np.mean((zero_long_logits>0)==long_y)),'history':zero_history,'first_validation_sample_initial_forget_bias1':gate_summary(initial_bias1_forget),'first_validation_sample_initial_forget_bias0':gate_summary(initial_bias0_forget),'first_validation_sample_learned_forget_bias1':gate_summary(learned_bias1_forget),'first_validation_sample_learned_forget_bias0':gate_summary(learned_bias0_forget)}}
print('REHEARSAL_METRICS='+json.dumps(metrics,ensure_ascii=False))
