import json,time

# 한 출력의 미분 기여를 각 입력 패치와 공유 필터에 다시 더합니다.
# 반환 계약은 dX, dK, db이며 원래 입력·필터·편향의 shape를 따릅니다.
def my_conv_backward(dY, cache):
    shape, filters, patch_rows, stride, padding = cache
    N,C,H,W=shape
    F,_,kh,kw=filters.shape
    oh,ow=dY.shape[2:]
    patches=patch_rows.reshape(N,oh,ow,C,kh,kw)
    input_gradient_padded=np.zeros((N,C,H+2*padding,W+2*padding))
    filter_gradient=np.zeros_like(filters)
    bias_gradient=np.zeros(F)
    for n in range(N):
        for f in range(F):
            for row in range(oh):
                for col in range(ow):
                    weight=dY[n,f,row,col]
                    bias_gradient[f]+=weight
                    filter_gradient[f]+=weight*patches[n,row,col]
                    r,c=row*stride,col*stride
                    input_gradient_padded[n,:,r:r+kh,c:c+kw]+=weight*filters[f]
    return input_gradient_padded[:,:,padding:padding+H,padding:padding+W],filter_gradient,bias_gradient

my_conv_errors=check_conv_backward(my_conv_backward)

# 입력 픽셀이 겹친 패치에 여러 번 등장하면 경로들의 합이 필요합니다.
my_coverage,my_filter_grad,my_bias_grad=my_conv_backward(np.ones_like(ones_Y),ones_cache)
np.testing.assert_allclose(my_coverage[0,0],[[1,2,1],[2,4,2],[1,2,1]])
assert my_coverage.shape==ones_X.shape and my_filter_grad.shape==ones_K.shape and my_bias_grad.shape==(1,)

# 평균 pooling은 각 창의 입력 네 곳에 같은 dY/4를 돌려줍니다. 반환값은 dX 하나입니다.
def my_mean_pool_backward(dY,cache):
    shape,mode,indices=cache
    assert mode=='mean'
    N,C,H,W=shape
    dX=np.zeros(shape)
    for row in range(H//2):
        for col in range(W//2):
            dX[:,:,2*row:2*row+2,2*col:2*col+2]=dY[:,:,row,col,None,None]/4
    return dX
pool_message=check_mean_pool_backward(my_mean_pool_backward)

# 의도적 오답: 새 패치의 기울기로 덮어 써서 앞 패치의 기여를 지웁니다.
def wrong_conv_overwrites(dY,cache):
    shape,filters,_,stride,padding=cache
    N,C,H,W=shape
    kh,kw=filters.shape[-2:]
    _,dK,db=my_conv_backward(dY,cache)
    bad_padded=np.zeros((N,C,H+2*padding,W+2*padding))
    for n in range(N):
        for row in range(dY.shape[2]):
            for col in range(dY.shape[3]):
                contribution=np.zeros((C,kh,kw))
                for f in range(filters.shape[0]):
                    contribution+=dY[n,f,row,col]*filters[f]
                r,c=row*stride,col*stride
                bad_padded[n,:,r:r+kh,c:c+kw]=contribution
    return bad_padded[:,:,padding:padding+H,padding:padding+W],dK,db

def wrong_mean_pool(dY,cache):
    return 4*my_mean_pool_backward(dY,cache)
rejections={}
for name,check,candidate in [('conv_overlap_overwrite',check_conv_backward,wrong_conv_overwrites),('pool_divisor_missing',check_mean_pool_backward,wrong_mean_pool)]:
    try:
        check(candidate)
    except AssertionError as exc:
        rejections[name]=str(exc)
    assert name in rejections,name+' 오답이 검출되지 않았습니다.'
bad_coverage=wrong_conv_overwrites(np.ones_like(ones_Y),ones_cache)[0]
np.testing.assert_allclose(bad_coverage[0,0],np.ones((3,3)))

# 본문 7절 변형: 필터 3개 -> pooled (N,3,3,3) -> head 입력 27.
# seed, 학습/검증 데이터, mini-batch 순서와 600회 업데이트를 유지합니다.
variation_rng=np.random.default_rng(503)
variation_p={'K':variation_rng.normal(size=(3,1,3,3))*np.sqrt(2/9),
             'c':np.zeros(3),'W':variation_rng.normal(size=(27,10))*np.sqrt(1/27),'b':np.zeros(10)}
assert variation_p['K'].shape==(3,1,3,3) and variation_p['W'].shape==(27,10)
parameter_count=sum(v.size for v in variation_p.values())
assert parameter_count==310
variation_state={key:(np.zeros_like(value),np.zeros_like(value)) for key,value in variation_p.items()}
variation_batch_rng=np.random.default_rng(505)
variation_history={'step':[],'train':[],'valid':[]}
variation_started=time.perf_counter()
for step in range(601):
    if step%50==0:
        train_scores,_=cnn_forward(X_train,variation_p)
        valid_scores,_=cnn_forward(X_valid,variation_p)
        variation_history['step'].append(step)
        variation_history['train'].append(float(ce_loss(train_scores,y_train)[0]))
        variation_history['valid'].append(float(ce_loss(valid_scores,y_valid)[0]))
    if step==600:
        break
    batch=variation_batch_rng.choice(len(X_train),50,replace=False)
    scores,variation_cache=cnn_forward(X_train[batch],variation_p)
    _,upstream=ce_loss(scores,y_train[batch])
    _,variation_gradients=cnn_backward(upstream,variation_cache,variation_p)
    adam_update(variation_p,variation_gradients,variation_state,step+1)
variation_seconds=time.perf_counter()-variation_started
variation_accuracy=float(np.mean(valid_scores.argmax(axis=1)==y_valid))
plot_curves(history['step'],{'필터 6개 검증 CE':history['valid'],'필터 3개 검증 CE':variation_history['valid']},title='직접 변형: 같은 600회 갱신에서 필터 수 변경',ylabel='CE')
metrics={'independent_conv_errors':{k:float(v) for k,v in my_conv_errors.items()},'independent_mean_pool_check':pool_message,'return_shapes':{'dX':list(my_coverage.shape),'dK':list(my_filter_grad.shape),'db':list(my_bias_grad.shape)},'overlap_coverage':my_coverage[0,0].tolist(),'wrong_overlap_coverage':bad_coverage[0,0].tolist(),'intentional_wrong_assertions':rejections,'baseline':{'train_ce_initial':float(history['train'][0]),'train_ce_final':float(history['train'][-1]),'validation_ce':float(history['valid'][-1]),'validation_accuracy':float(valid_accuracy),'parameters':sum(v.size for v in p.values()),'conv_gradient_max_error':float(max(conv_errors.values())),'cnn_gradient_max_error':float(max(cnn_errors.values()))},'filter3_variation':{'head_input_dimension':27,'parameters':parameter_count,'steps':600,'train_ce_final':variation_history['train'][-1],'validation_ce':variation_history['valid'][-1],'validation_accuracy':variation_accuracy,'training_seconds':variation_seconds,'history':variation_history}}
print('REHEARSAL_METRICS='+json.dumps(metrics,ensure_ascii=False))
