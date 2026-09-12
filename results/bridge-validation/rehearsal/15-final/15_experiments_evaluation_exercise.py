import json

# 관측별 혼동행렬, 정답 확률의 로그, 양성·음성 쌍의 순위를 별도 구현합니다.
# 원본 binary_metrics는 호출하지 않습니다.
def my_binary_metrics(y,p,threshold=.5):
    y,p=np.asarray(y),np.asarray(p)
    if y.ndim!=1 or p.shape!=y.shape or len(y)==0:
        raise ValueError('같은 길이의 비어 있지 않은 1차원 배열을 사용합니다.')
    if not np.all(np.isin(y,[0,1])) or not np.all(np.isfinite(p)) or np.any((p<0)|(p>1)):
        raise ValueError('y는 0/1, p는 0과 1 사이의 유한한 확률입니다.')
    confusion=np.zeros((2,2),dtype=int)
    nll_sum,brier_sum=0.,0.
    positive_scores,negative_scores=[],[]
    for label,probability in zip(y,p):
        predicted=1 if probability>=threshold else 0
        confusion[int(label),predicted]+=1
        correct_probability=probability if label==1 else 1-probability
        if correct_probability==0:
            nll_sum=np.inf
        else:
            nll_sum-=np.log(correct_probability)
        brier_sum+=(probability-label)**2
        if label==1:
            positive_scores.append(probability)
        else:
            negative_scores.append(probability)
    tn,fp=confusion[0]
    fn,tp=confusion[1]
    pair_score=0.
    for positive in positive_scores:
        for negative in negative_scores:
            if positive>negative:
                pair_score+=1.
            elif positive==negative:
                pair_score+=.5
    pairs=len(positive_scores)*len(negative_scores)
    return {'accuracy':(tp+tn)/len(y),
            'precision':tp/(tp+fp) if tp+fp else 0.,
            'recall':tp/(tp+fn) if tp+fn else 0.,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            'nll':nll_sum/len(y),'brier':brier_sum/len(y),
            'auc':pair_score/pairs if pairs else np.nan,'confusion':confusion}

checker_result=check_binary_metrics(my_binary_metrics)
print('직접 쓴 지표:',checker_result)

# 임계값과 같은 확률 및 AUC 동점을 손계산합니다.
my_hand_y=np.array([0,1,0,1]); my_hand_p=np.array([.2,.5,.5,.8])
my_hand=my_binary_metrics(my_hand_y,my_hand_p,.5)
np.testing.assert_array_equal(my_hand['confusion'],[[1,1],[0,2]])
np.testing.assert_allclose([my_hand['accuracy'],my_hand['precision'],my_hand['recall'],my_hand['f1'],my_hand['auc'],my_hand['brier']],
                           [.75,2/3,1.,.8,.875,.145])
np.testing.assert_allclose(my_hand['nll'],-.5*np.log(.4))
my_no_positive=my_binary_metrics([0,1,1],[.1,.2,.3],.5)
np.testing.assert_allclose([my_no_positive['precision'],my_no_positive['recall'],my_no_positive['f1']],[0.,0.,0.])
my_single_negative=my_binary_metrics([0,0,0],[.1,.2,.8])
my_single_positive=my_binary_metrics([1,1],[.1,.2])
assert np.isnan(my_single_negative['auc']) and np.isnan(my_single_positive['auc'])
my_perfect=my_binary_metrics([0,1],[0.,1.])
my_wrong_certain=my_binary_metrics([0,1],[1.,0.])
assert my_perfect['nll']==0 and np.isinf(my_wrong_certain['nll'])

# 의도적 오답: >= 대신 >, recall 분모, AUC 동점, 단일 클래스, NLL 확률 0 처리.
def wrong_strict_threshold(y,p,threshold=.5):
    return my_binary_metrics(y,p,np.nextafter(float(threshold),np.inf))


def wrong_recall_denominator(y,p,threshold=.5):
    result=my_binary_metrics(y,p,threshold)
    result['recall']=result['precision']
    return result


def wrong_auc_ties(y,p,threshold=.5):
    result=my_binary_metrics(y,p,threshold)
    positives=np.asarray(p)[np.asarray(y)==1]
    negatives=np.asarray(p)[np.asarray(y)==0]
    if len(positives) and len(negatives):
        result['auc']=np.mean(positives[:,None]>negatives[None,:])
    return result


def wrong_single_class_auc(y,p,threshold=.5):
    result=my_binary_metrics(y,p,threshold)
    if np.isnan(result['auc']): result['auc']=0.
    return result


def wrong_clip_nll(y,p,threshold=.5):
    result=my_binary_metrics(y,p,threshold)
    cp=np.where(np.asarray(y)==1,p,1-np.asarray(p))
    result['nll']=-np.log(np.clip(cp,1e-12,1)).mean()
    return result

negative_results={}
for name,candidate in [('strict_threshold',wrong_strict_threshold),('wrong_recall_denominator',wrong_recall_denominator),
                       ('ignores_auc_ties',wrong_auc_ties),('single_class_auc_zero',wrong_single_class_auc),('clips_zero_correct_probability',wrong_clip_nll)]:
    try:
        result=check_binary_metrics(candidate)
    except AssertionError as exc:
        negative_results[name]={'rejected':True,'error_type':type(exc).__name__,'message':str(exc).strip()}
    else:
        negative_results[name]={'rejected':False,'outcome':result}
assert all(item['rejected'] for item in negative_results.values())

# 본문 변형: 절편만 -2.5로 바꾸고 데이터 생성과 분할부터 새 실험을 수행합니다.
# 선택 전에는 test 입력의 변환이나 test 지표 계산을 하지 않습니다.
my_data_rng=np.random.default_rng(151)
my_split_rng=np.random.default_rng(152)
my_raw_X=my_data_rng.normal(size=(900,8))*np.array([3.,.5,2.,1.,.2,4.,1.,2.])+np.arange(8)
my_true_logits=1.2*(my_raw_X[:,0]/3)-1.5*((my_raw_X[:,1]-1)/.5)-2.5
my_raw_y=(my_data_rng.random(900)<sigmoid(my_true_logits)).astype(float)
my_order=my_split_rng.permutation(900)
my_train_index,my_valid_index,my_test_index=my_order[:540],my_order[540:720],my_order[720:]
my_overlap_counts=[len(set(my_train_index)&set(my_valid_index)),len(set(my_train_index)&set(my_test_index)),len(set(my_valid_index)&set(my_test_index))]
assert my_overlap_counts==[0,0,0]
np.testing.assert_array_equal(np.sort(my_order),np.arange(900))
my_mean,my_scale=fit_standardizer(my_raw_X[my_train_index])
np.testing.assert_allclose(my_mean,my_raw_X[my_train_index].mean(axis=0))
my_X_train=transform(my_raw_X[my_train_index],my_mean,my_scale)
my_X_valid=transform(my_raw_X[my_valid_index],my_mean,my_scale)
my_y_train,my_y_valid=my_raw_y[my_train_index],my_raw_y[my_valid_index]
my_seeds=[1,2,3]; my_l2_values=[0.,.02,.2]
my_runs={l2:[train_logistic(my_X_train,my_y_train,my_X_valid,my_y_valid,l2=l2,seed=s) for s in my_seeds] for l2 in my_l2_values}
my_mean_validation={l2:float(np.mean([r['validation_bce'] for r in group])) for l2,group in my_runs.items()}
my_selected_l2=min(my_mean_validation,key=my_mean_validation.get)
my_selected_runs=my_runs[my_selected_l2]
my_snapshot_errors=[]
for run in my_selected_runs:
    snapshot_validation=logistic_objective(my_X_valid,my_y_valid,run['w'],run['b'])[1]
    snapshot_training=logistic_objective(my_X_train,my_y_train,run['w'],run['b'],my_selected_l2)
    chosen_history=run['history'][run['epoch']-1]
    np.testing.assert_allclose(snapshot_validation,run['validation_bce'],atol=1e-14)
    np.testing.assert_allclose([snapshot_training[1],snapshot_validation,snapshot_training[0]],chosen_history[:3],atol=1e-14)
    my_snapshot_errors.append(float(abs(snapshot_validation-run['validation_bce'])))
my_valid_probabilities=[sigmoid(my_X_valid @ run['w']+run['b']) for run in my_selected_runs]
my_mean_valid_p=np.mean(my_valid_probabilities,axis=0)
my_thresholds=np.linspace(.1,.9,33)
my_threshold_f1=np.array([my_binary_metrics(my_y_valid,my_mean_valid_p,t)['f1'] for t in my_thresholds])
my_best_threshold_index=int(np.argmax(my_threshold_f1))
my_selected_threshold=float(my_thresholds[my_best_threshold_index])
my_threshold_ties=my_thresholds[my_threshold_f1==my_threshold_f1.max()].tolist()
my_frozen_choice={'l2':my_selected_l2,'epochs':[r['epoch'] for r in my_selected_runs],
                  'threshold':my_selected_threshold,'training_seeds':my_seeds.copy(),
                  'selection':'seed mean validation BCE, then ensemble validation F1'}

# 선택을 고정한 뒤 학습 통계로 test를 변환하고 공통 임계값으로 평가합니다.
my_X_test=transform(my_raw_X[my_test_index],my_mean,my_scale)
my_y_test=my_raw_y[my_test_index]
my_test_probabilities=[sigmoid(my_X_test @ r['w']+r['b']) for r in my_selected_runs]
my_test_results=[my_binary_metrics(my_y_test,p,my_selected_threshold) for p in my_test_probabilities]
my_ensemble_p=np.mean(my_test_probabilities,axis=0)
my_test_ensemble=my_binary_metrics(my_y_test,my_ensemble_p,my_selected_threshold)
my_test_nll=np.array([m['nll'] for m in my_test_results])
my_always_negative=my_binary_metrics(my_y_test,np.zeros_like(my_y_test))
np.testing.assert_allclose(my_always_negative['accuracy'],1-my_y_test.mean())
assert my_always_negative['recall']==0
assert my_frozen_choice['threshold']==my_selected_threshold and my_frozen_choice['l2']==my_selected_l2

plot_curves(my_thresholds,{'새 validation F1':my_threshold_f1},
            title='절편 -2.5: 검증으로 다시 선택한 임계값',xlabel='양성 판정 임계값',ylabel='F1')
plot_matrices([ensemble_result['confusion'],my_test_ensemble['confusion']],
              ['기존 test: 절편 -1.2','새 test: 절편 -2.5'])

# JSON에 정의되지 않은 실수는 문자열로 표시하여 표준 JSON을 유지합니다.
def json_safe(value):
    if isinstance(value,dict): return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [json_safe(v) for v in value]
    if isinstance(value,np.ndarray): return json_safe(value.tolist())
    if isinstance(value,np.generic): return json_safe(value.item())
    if isinstance(value,float):
        if np.isnan(value): return 'NaN'
        if np.isposinf(value): return '+Infinity'
        if np.isneginf(value): return '-Infinity'
    return value

rehearsal_metrics={
 'checker_result':checker_result,'hand_threshold_and_auc_tie':my_hand,
 'edge_cases':{'no_positive_prediction':my_no_positive,'all_negative_labels':my_single_negative,
                'all_positive_labels':my_single_positive,'perfect_endpoints':my_perfect,'wrong_certain_endpoints':my_wrong_certain},
 'negative_candidates':negative_results,
 'changed_experiment':{'intercept':-2.5,'data_seed':151,'split_seed':152,'counts':[540,180,180],
                       'split_overlap_counts':my_overlap_counts,'training_seeds':my_seeds,'epochs_per_run':120,
                       'batch_size':64,'last_batch_size':540%64,'training_run_count':9,
                       'train_positive_fraction':float(my_y_train.mean()),'validation_positive_fraction':float(my_y_valid.mean()),
                       'test_positive_fraction':float(my_y_test.mean()),'mean_validation_bce':my_mean_validation,
                       'frozen_choice_before_test':my_frozen_choice,'snapshot_validation_errors':my_snapshot_errors,
                       'thresholds_tied_for_max_validation_f1':my_threshold_ties,'max_validation_f1':float(my_threshold_f1.max()),
                       'ensemble_test':my_test_ensemble,'individual_test':my_test_results,
                       'test_seed_nll_mean':float(my_test_nll.mean()),'test_seed_nll_sample_std':float(my_test_nll.std(ddof=1)),
                       'always_negative_accuracy':my_always_negative['accuracy'],'always_negative_recall':my_always_negative['recall'],
                       'train_mean':my_mean,'train_scale':my_scale,
                       'full_data_mean_difference':float(np.max(np.abs(my_mean-my_raw_X.mean(axis=0))))},
 'original_experiment':{'intercept':-1.2,'train_positive_fraction':float(y_train.mean()),'validation_positive_fraction':float(y_valid.mean()),
                        'test_positive_fraction':float(y_test.mean()),'ensemble_test':ensemble_result,'experiment_record':experiment_record,
                        'mean_validation_bce':mean_validation,'row_split_accuracy':float(row_accuracy),'group_split_accuracy':float(group_accuracy),
                        'row_overlap_people':row_overlap,'gradient_errors':check_errors,'tiny_bce':float(tiny['validation_bce']),
                        'ordinary_validation':ordinary_metrics,'sharp_validation':sharp_metrics,'validation_ece':float(ece),
                        'permuted_validation_nll':permuted_nll,'test_seed_nll_mean':float(seed_nll.mean()),
                        'test_seed_nll_sample_std':float(seed_nll.std(ddof=1))},
 'nonfinite_encoding':{'NaN':'한 클래스의 ROC-AUC처럼 정의되지 않은 값','+Infinity':'정답 확률 0의 NLL'},
}
print('REHEARSAL_METRICS='+json.dumps(json_safe(rehearsal_metrics),ensure_ascii=False,allow_nan=False))
