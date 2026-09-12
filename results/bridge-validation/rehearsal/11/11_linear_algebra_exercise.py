import json

# mu_j=sum_n X_nj/N, C_ij=sum_n (X_ni-mu_i)(X_nj-mu_j)/N.
# eigh의 고유값은 오름차순이므로 열 순서를 뒤집고 큰 k개 방향을 취합니다.
# 반환값 mu:(D,), Q:(D,k). 원본 pca_fit/covariance를 호출하지 않습니다.
def my_pca_fit(X,k):
    N,D=X.shape
    if not 1<=k<=min(N,D):
        raise ValueError('k 범위를 확인하세요.')
    average=np.sum(X,axis=0)/N
    covariance_matrix=np.zeros((D,D))
    for point in X:
        centered_point=point-average
        covariance_matrix+=centered_point[:,None]*centered_point[None,:]/N
    variances,directions=np.linalg.eigh(covariance_matrix)
    chosen=directions[:,::-1][:,:k]
    return average,chosen

candidate_message=check_pca_fit(my_pca_fit)

# 오답 1: 같은 mean을 반환하지만 방향은 중심화하지 않은 2차 모멘트에서 찾습니다.
def wrong_uncentered(X,k):
    average=np.mean(X,axis=0)
    _,directions=np.linalg.eigh(X.T@X/len(X))
    return average,directions[:,::-1][:,:k]

# 오답 2: 오름차순에서 처음 k개를 골라 가장 작은 분산을 남깁니다.
def wrong_smallest_variance(X,k):
    average=np.mean(X,axis=0)
    centered=X-average
    _,directions=np.linalg.eigh(centered.T@centered/len(X))
    return average,directions[:,:k]

# 오답 3: 열에 방향을 두어야 하는 반환 계약을 뒤집습니다.
def wrong_transposed_Q(X,k):
    average,directions=my_pca_fit(X,k)
    return average,directions.T

rejections={}
for name,candidate in [('uncentered_second_moment',wrong_uncentered),('smallest_variance',wrong_smallest_variance),('transposed_Q',wrong_transposed_Q)]:
    try:
        check_pca_fit(candidate)
    except AssertionError as exc:
        rejections[name]={'type':type(exc).__name__,'message':str(exc),'repr':repr(exc)}
    assert name in rejections,name+' 오답을 검사기가 검출하지 못했습니다.'

# 독립적인 3차원 입력을 만들어 같은 학습 mean/Q로 새 검증 입력을 처리합니다.
new_rng=np.random.default_rng(1117)
new_transform=np.array([[2.,0.3,-0.2],[0.,0.8,0.5],[0.,0.,0.15]])
new_shift=np.array([4.,-2.,1.])
new_train=new_rng.normal(size=(80,3))@new_transform+new_shift
new_valid=new_rng.normal(size=(45,3))@new_transform+new_shift
new_train_errors=[]
new_valid_errors=[]
new_shapes=[]
new_tail_variance=[]
new_eigenvalues=np.linalg.eigvalsh((new_train-new_train.mean(axis=0)).T@(new_train-new_train.mean(axis=0))/len(new_train))
for kept in [1,2,3]:
    fitted_mean,fitted_Q=my_pca_fit(new_train,kept)
    train_scores=(new_train-fitted_mean)@fitted_Q
    train_rebuilt=train_scores@fitted_Q.T+fitted_mean
    valid_scores=(new_valid-fitted_mean)@fitted_Q
    valid_rebuilt=valid_scores@fitted_Q.T+fitted_mean
    train_error=float(np.mean(np.sum((new_train-train_rebuilt)**2,axis=1)))
    valid_error=float(np.mean(np.sum((new_valid-valid_rebuilt)**2,axis=1)))
    tail=float(np.sum(new_eigenvalues[:3-kept]))
    np.testing.assert_allclose(train_error,tail,atol=1e-12)
    np.testing.assert_allclose(train_error,3*np.mean((new_train-train_rebuilt)**2),atol=1e-12)
    np.testing.assert_allclose((new_valid-valid_rebuilt)@fitted_Q,0,atol=1e-12)
    new_train_errors.append(train_error)
    new_valid_errors.append(valid_error)
    new_tail_variance.append(tail)
    new_shapes.append({'k':kept,'mu':list(fitted_mean.shape),'Q':list(fitted_Q.shape),'Z_train':list(train_scores.shape),'Z_valid':list(valid_scores.shape),'reconstruction_valid':list(valid_rebuilt.shape)})
assert np.all(np.diff(new_valid_errors)<=1e-12)
np.testing.assert_allclose(valid_rebuilt,new_valid,atol=1e-12)

# 앞에서 학습한 k=2 기저로 손에 잡히는 새로운 점 세 개를 투영합니다.
novel_points=np.array([[6.,-1.5,0.5],[2.,-3.,1.7],[4.,-1.8,1.2]])
novel_mean,novel_Q=my_pca_fit(new_train,2)
novel_Z=(novel_points-novel_mean)@novel_Q
novel_reconstructed=novel_Z@novel_Q.T+novel_mean
np.testing.assert_allclose((novel_points-novel_reconstructed)@novel_Q,0,atol=1e-12)
flipped_Q=-novel_Q
flipped_Z=(novel_points-novel_mean)@flipped_Q
flipped_reconstruction=flipped_Z@flipped_Q.T+novel_mean
np.testing.assert_allclose(flipped_reconstruction,novel_reconstructed,atol=1e-12)
np.testing.assert_allclose(flipped_Z,-novel_Z,atol=1e-12)

plot_curves([1,2,3],{'직접 만든 학습 입력':new_train_errors,'직접 만든 검증 입력':new_valid_errors},title='독립 PCA 구현: 새 3차원 입력의 복원 거리',xlabel='남긴 방향 k',ylabel='관측별 평균 제곱 거리')
plot_matrices([novel_points,novel_Z,novel_reconstructed],['새 입력 (3,3)','압축 좌표 (3,2)','학습한 mean/Q로 복원 (3,3)'])

# 가장 가까운 점의 손계산과 경험 공분산을 다시 숫자로 대조합니다.
manual_projection=np.array([[1.,1.]])
np.testing.assert_allclose(project(np.array([[2.,0.]]),np.array([[1.],[1.]])/np.sqrt(2))[1],manual_projection)
np.testing.assert_allclose(C,[[8/3,4/3],[4/3,8/3]])
projection_manual_gradient=np.array([2.,0.])-manual_projection[0]
np.testing.assert_allclose(projection_manual_gradient@np.array([1.,1.]),0.)
metrics={'independent_pca_checker':candidate_message,'intentional_wrong_rejections':rejections,'baseline':{'retained_variance_fraction':float(retained_fraction),'train_mean_squared_distance':float(train_distance),'valid_mean_squared_distance':float(valid_distance),'projection_backward_max_relative_error':float(max(error_x,error_q)),'source_3d_validation_errors':[float(v) for v in rank_errors]},'new_input_variation':{'seed':1117,'train_shape':list(new_train.shape),'valid_shape':list(new_valid.shape),'shapes':new_shapes,'train_errors':new_train_errors,'validation_errors':new_valid_errors,'discarded_covariance_eigenvalues_sum':new_tail_variance,'all_training_distances_match_discarded_variance':True,'element_mse_times_D_matches_distance':True,'novel_points':novel_points.tolist(),'trained_mean_for_new_points':novel_mean.tolist(),'trained_Q_for_new_points':novel_Q.tolist(),'novel_Z':novel_Z.tolist(),'novel_reconstruction':novel_reconstructed.tolist(),'novel_mean_squared_distance':float(np.mean(np.sum((novel_points-novel_reconstructed)**2,axis=1))),'basis_sign_flip_reconstruction_max_abs_error':float(np.max(np.abs(flipped_reconstruction-novel_reconstructed)))}}
print('REHEARSAL_METRICS='+json.dumps(metrics,ensure_ascii=False))
