import json,time

# 토큰 하나의 Jacobian을 수식으로 만듭니다.
# J_jk = inverse * (1[j=k] - 1/D - xhat_j*xhat_k/D).
# dX=J.T@(dY*gamma), dgamma와 dbeta는 토큰·배치에서 공유된 기여의 합입니다.
def my_layernorm_backward(dY,cache):
    normalized,inverse,gamma=cache
    N,T,D=dY.shape
    dX=np.zeros_like(dY)
    dgamma=np.zeros(D)
    dbeta=np.zeros(D)
    for n in range(N):
        for t in range(T):
            xhat=normalized[n,t]
            jacobian=inverse[n,t,0]*(np.eye(D)-np.ones((D,D))/D-np.outer(xhat,xhat)/D)
            dX[n,t]=jacobian.T@(dY[n,t]*gamma)
            dgamma+=dY[n,t]*xhat
            dbeta+=dY[n,t]
    return dX,dgamma,dbeta

my_ln_errors=check_layernorm(my_layernorm_backward)

# 같은 dY가 잔차의 항등 경로와 FFN 경로 양쪽으로 출발합니다.
# 이어서 dU도 첫 잔차의 항등 경로와 MHA 경로 양쪽으로 출발합니다.
def my_block_backward(dY,cache,p):
    first_norm_cache,attention_cache,second_norm_cache,feedforward_cache=cache
    d_normalized_u,ffn_gradients=ffn_backward(dY,feedforward_cache,p)
    d_u_via_ffn,dgamma2,dbeta2=my_layernorm_backward(d_normalized_u,second_norm_cache)
    d_u=dY.copy()+d_u_via_ffn
    d_normalized_x,attention_gradients=mha_backward(d_u,attention_cache,p)
    d_x_via_attention,dgamma1,dbeta1=my_layernorm_backward(d_normalized_x,first_norm_cache)
    d_x=d_u.copy()+d_x_via_attention
    gradients={**ffn_gradients,**attention_gradients,'gamma1':dgamma1,'beta1':dbeta1,'gamma2':dgamma2,'beta2':dbeta2}
    return d_x,gradients

my_block_errors=check_block_backward(my_block_backward)
my_transformer_errors=check_transformer(my_block_backward)

# 원래 잔차 항등 예제를 독립 block 후보에 직접 전달합니다.
my_identity_dx,_=my_block_backward(identity_G,identity_cache,identity_p)
np.testing.assert_allclose(my_identity_dx,identity_G)
residual_identity_error=float(np.max(np.abs(my_identity_dx-identity_G)))
residual_rejection=None
try:
    check_block_backward(wrong_missing_residual)
except AssertionError as exc:
    residual_rejection=str(exc)
assert residual_rejection is not None

# N=A=2인 경우에도 배치 축을 head 축으로 오해하지 않아야 합니다.
mask_rng=np.random.default_rng(909)
mask_X=mask_rng.normal(size=(2,3,4))
mask_p={key:mask_rng.normal(size=(4,4))*0.3 for key in ['Wq','Wk','Wv','Wo']}
shared_mask=np.tril(np.ones((3,3),dtype=bool))
batch_mask=np.stack([shared_mask,np.eye(3,dtype=bool)])
mask_rejection=None
try:
    mha_forward(mask_X,mask_p,heads=2,allowed=batch_mask)
except ValueError as exc:
    mask_rejection=str(exc)
assert mask_rejection is not None
batch_output,batch_cache=mha_forward(mask_X,mask_p,heads=2,allowed=batch_mask[:,None])
batch_weights=batch_cache[2][3]
for n in range(2):
    single_output,_=mha_forward(mask_X[n:n+1],mask_p,heads=2,allowed=batch_mask[n])
    np.testing.assert_allclose(batch_output[n:n+1],single_output)
    for a in range(2):
        assert np.all(batch_weights[n,a][~batch_mask[n]]==0)
np.testing.assert_allclose(batch_weights.sum(axis=-1),1.)

# head별 mask도 head 축을 명시해 위치를 구분합니다.
head_masks=np.repeat(batch_mask[:,None],2,axis=1)
head_masks[0,1]=np.ones((3,3),dtype=bool)
head_masks[1,1]=np.fliplr(np.eye(3,dtype=bool))
head_mask_output,head_mask_cache=mha_forward(mask_X,mask_p,heads=2,allowed=head_masks)
head_mask_weights=head_mask_cache[2][3]
assert head_mask_weights.shape==(2,2,3,3)
assert np.all(head_mask_weights[~head_masks]==0)
np.testing.assert_allclose(head_mask_weights.sum(axis=-1),1.)
np.testing.assert_allclose(head_mask_weights[1,1],np.fliplr(np.eye(3)))
assert head_mask_weights[0,1,0,1]>0 and head_mask_weights[0,0,0,1]==0

# embedding 표에 같은 token=1이 세 번 등장합니다.
repeated_ids=np.array([[1,0,1],[2,1,0]])
contribution_grid=np.array([[[1.,2.],[5.,-1.],[3.,4.]],[[-2.,2.],[-1.,1.],[2.,-3.]]])
manual_embedding=np.zeros((3,2))
for n in range(repeated_ids.shape[0]):
    for t in range(repeated_ids.shape[1]):
        manual_embedding[repeated_ids[n,t]]+=contribution_grid[n,t]
np.testing.assert_allclose(manual_embedding,[[7,-4],[3,7],[-2,2]])
add_at_embedding=np.zeros((3,2))
np.add.at(add_at_embedding,repeated_ids,contribution_grid)
np.testing.assert_array_equal(add_at_embedding,manual_embedding)
bad_indexed_embedding=np.zeros((3,2))
bad_indexed_embedding[repeated_ids]+=contribution_grid
assert not np.array_equal(bad_indexed_embedding,manual_embedding)

# D=12를 고정한 head 4개: (N,T,12)->(N,4,T,3), 점수 분모 sqrt(3).
head4_probe=mask_rng.normal(size=(2,4,12))
head4_split=split_heads(head4_probe,4)
assert head4_split.shape==(2,4,4,3)
np.testing.assert_array_equal(merge_heads(head4_split),head4_probe)
unsupported_head_rejected=False
try:
    split_heads(head4_probe,5)
except AssertionError:
    unsupported_head_rejected=True
assert unsupported_head_rejected

# 본문 학습의 초기값·데이터·배치 seed·800스텝을 유지하고 heads만 4로 전달합니다.
variation_p=init_transformer()
variation_parameter_count=sum(value.size for value in variation_p.values())
assert variation_parameter_count==sum(value.size for value in p.values())
variation_state={key:(np.zeros_like(value),np.zeros_like(value)) for key,value in variation_p.items()}
variation_rng=np.random.default_rng(902)
variation_history={'step':[],'train':[],'valid':[]}
variation_started=time.perf_counter()
for step in range(801):
    if step%80==0:
        variation_train_logits,_=transformer_forward(train_tokens,variation_p,heads=4)
        variation_valid_logits,_=transformer_forward(valid_tokens,variation_p,heads=4)
        variation_history['step'].append(step)
        variation_history['train'].append(float(token_ce(variation_train_logits,train_targets)[0]))
        variation_history['valid'].append(float(token_ce(variation_valid_logits,valid_targets)[0]))
    if step==800:
        break
    batch=variation_rng.choice(len(train_tokens),64,replace=False)
    scores,variation_cache=transformer_forward(train_tokens[batch],variation_p,heads=4)
    _,upstream=token_ce(scores,train_targets[batch])
    gradients=transformer_backward(upstream,variation_cache,variation_p)
    norm=np.sqrt(sum(np.sum(g*g) for g in gradients.values()))
    scale=min(1.,1./(norm+1e-12))
    adam_step(variation_p,{key:value*scale for key,value in gradients.items()},variation_state,step+1)
variation_seconds=time.perf_counter()-variation_started
variation_predictions=variation_valid_logits.argmax(axis=-1)
variation_token_accuracy=float(np.mean(variation_predictions==valid_targets))
variation_sequence_accuracy=float(np.mean(np.all(variation_predictions==valid_targets,axis=1)))
_,variation_trained_cache=transformer_forward(valid_tokens[:1],variation_p,heads=4)
variation_attention=variation_trained_cache[2][1][2][3][0]
assert variation_attention.shape==(4,4,4)
plot_curves(history['step'],{'head 2개 검증 CE':history['valid'],'head 4개 검증 CE':variation_history['valid']},title='직접 head 변형: D=12와 800회 갱신 유지',ylabel='토큰 평균 CE')
plot_matrices([variation_attention[h] for h in range(4)],[f'직접 변형 head {h}: query→key' for h in range(4)],cmap='Blues')
metrics={'independent_layernorm_errors':{k:float(v) for k,v in my_ln_errors.items()},'independent_block_errors':{k:float(v) for k,v in my_block_errors.items()},'independent_transformer_errors':{k:float(v) for k,v in my_transformer_errors.items()},'residual_identity_max_abs_error':residual_identity_error,'missing_residual_rejection':residual_rejection,'mha_masks':{'wrong_3d_shape':[2,3,3],'wrong_3d_rejection':mask_rejection,'correct_batch_shape':list(batch_mask[:,None].shape),'attention_weights_shape':list(batch_weights.shape),'batch_mask_weights':batch_weights.tolist(),'explicit_head_mask_shape':list(head_masks.shape),'explicit_head_mask_weights':head_mask_weights.tolist(),'separate_batch_forward_matches':True,'masked_weights_all_zero':True},'embedding':{'ids':repeated_ids.tolist(),'manual_gradient':manual_embedding.tolist(),'add_at_gradient':add_at_embedding.tolist(),'wrong_advanced_index_gradient':bad_indexed_embedding.tolist(),'wrong_advanced_index_detected':True},'baseline_heads2':{'parameters':sum(v.size for v in p.values()),'train_ce_initial':float(history['train'][0]),'train_ce_final':float(history['train'][-1]),'validation_ce':float(history['valid'][-1]),'token_accuracy':float(token_accuracy),'sequence_accuracy':float(sequence_accuracy)},'heads4_variation':{'heads':4,'model_width':12,'head_width':3,'score_denominator':float(np.sqrt(3)),'split_shape':list(head4_split.shape),'parameters':variation_parameter_count,'projection_parameters':sum(variation_p[key].size for key in ['Wq','Wk','Wv','Wo']),'steps':800,'training_seconds':variation_seconds,'train_ce_initial':variation_history['train'][0],'train_ce_final':variation_history['train'][-1],'validation_ce':variation_history['valid'][-1],'token_accuracy':variation_token_accuracy,'sequence_accuracy':variation_sequence_accuracy,'heads5_rejected':unsupported_head_rejected,'history':variation_history}}
print('REHEARSAL_METRICS='+json.dumps(metrics,ensure_ascii=False))
