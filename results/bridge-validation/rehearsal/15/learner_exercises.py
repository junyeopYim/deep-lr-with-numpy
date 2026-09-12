"""15 실습자가 독립 작성한 지표와 실행 후 검산입니다. 저장소를 쓰지 않습니다."""
import hashlib
import json
import math
from pathlib import Path


def my_binary_metrics(y, p, threshold=.5):
    y = np.asarray(y)
    p = np.asarray(p)
    if y.ndim != 1 or p.shape != y.shape or y.size == 0:
        raise ValueError('같은 길이의 비어 있지 않은 1차원 배열이 필요합니다.')
    if any(label not in (0, 1) for label in y):
        raise ValueError('라벨은 0 또는 1입니다.')
    if any(not math.isfinite(float(score)) or not 0 <= score <= 1 for score in p):
        raise ValueError('확률은 0 이상 1 이하의 유한한 값입니다.')
    tn = fp = fn = tp = 0
    log_costs = []
    square_cost = 0.
    positive_scores = []
    negative_scores = []
    for label, score in zip(y, p):
        decision = score >= threshold
        if label == 1:
            positive_scores.append(score)
            tp += int(decision)
            fn += int(not decision)
            correct_probability = score
        else:
            negative_scores.append(score)
            fp += int(decision)
            tn += int(not decision)
            correct_probability = 1 - score
        log_costs.append(-math.log(correct_probability) if correct_probability > 0 else math.inf)
        square_cost += float((score - label) ** 2)
    credit = 0.
    pair_count = 0
    for positive in positive_scores:
        for negative in negative_scores:
            pair_count += 1
            if positive > negative:
                credit += 1
            elif positive == negative:
                credit += .5
    return {
        'accuracy': (tp + tn) / y.size,
        'precision': tp / (tp + fp) if tp + fp else 0.,
        'recall': tp / (tp + fn) if tp + fn else 0.,
        'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
        'nll': sum(log_costs) / y.size,
        'brier': square_cost / y.size,
        'auc': credit / pair_count if pair_count else math.nan,
        'confusion': np.array([[tn, fp], [fn, tp]]),
    }


def _classification_counts(y, decisions):
    tn = fp = fn = tp = 0
    for label, decision in zip(y, decisions):
        tn += int(label == 0 and not decision)
        fp += int(label == 0 and decision)
        fn += int(label == 1 and not decision)
        tp += int(label == 1 and decision)
    return {
        'accuracy': (tn + tp) / len(y),
        'precision': tp / (tp + fp) if tp + fp else 0.,
        'recall': tp / (tp + fn) if tp + fn else 0.,
        'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
        'confusion': np.array([[tn, fp], [fn, tp]]),
    }


def wrong_strict_threshold(y, p, threshold=.5):
    result = my_binary_metrics(y, p, threshold)
    result.update(_classification_counts(y, p > threshold))
    return result


def wrong_f1_arithmetic_mean(y, p, threshold=.5):
    result = my_binary_metrics(y, p, threshold)
    result['f1'] = (result['precision'] + result['recall']) / 2
    return result


def wrong_single_class_auc_zero(y, p, threshold=.5):
    result = my_binary_metrics(y, p, threshold)
    if math.isnan(result['auc']):
        result['auc'] = 0.
    return result


def wrong_auc_without_tie_credit(y, p, threshold=.5):
    result = my_binary_metrics(y, p, threshold)
    pairs = [(pos, neg) for pos in p[y == 1] for neg in p[y == 0]]
    result['auc'] = sum(pos > neg for pos, neg in pairs) / len(pairs) if pairs else math.nan
    return result


def wrong_finite_nll_by_clipping(y, p, threshold=.5):
    result = my_binary_metrics(y, p, threshold)
    result['nll'] = my_binary_metrics(y, np.clip(p, 1e-12, 1-1e-12), threshold)['nll']
    return result


def wrong_confusion_transpose(y, p, threshold=.5):
    result = my_binary_metrics(y, p, threshold)
    result['confusion'] = result['confusion'].T
    return result


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def collect_rehearsal_metrics(label, intercept, source_hash, target_dir):
    checker_message = check_binary_metrics(my_binary_metrics)
    own_threshold_f1 = np.array([
        my_binary_metrics(y_valid, mean_validation_probability, threshold)['f1']
        for threshold in thresholds
    ])
    np.testing.assert_allclose(own_threshold_f1, threshold_f1, rtol=0, atol=1e-12)
    own_threshold = float(thresholds[np.argmax(own_threshold_f1)])
    assert own_threshold == selected_threshold
    own_test = my_binary_metrics(y_test, ensemble_probability, own_threshold)
    for key in ensemble_result:
        np.testing.assert_allclose(own_test[key], ensemble_result[key], rtol=0, atol=1e-12)

    split_groups = [train_index, valid_index, test_index]
    assert [len(indices) for indices in split_groups] == [540, 180, 180]
    np.testing.assert_array_equal(np.sort(np.concatenate(split_groups)), np.arange(900))
    assert not any(set(a) & set(b) for a, b in [(train_index, valid_index),
                                             (train_index, test_index),
                                             (valid_index, test_index)])
    np.testing.assert_allclose(train_mean, raw_X[train_index].mean(axis=0), rtol=0, atol=0)
    np.testing.assert_allclose(train_scale, raw_X[train_index].std(axis=0), rtol=0, atol=0)
    np.testing.assert_allclose(X_train.mean(axis=0), 0, atol=2e-13)
    np.testing.assert_allclose(X_train.std(axis=0), 1, atol=2e-13)
    changed_future_X = raw_X.copy()
    changed_future_X[test_index] += 1000000
    unchanged_mean, unchanged_scale = fit_standardizer(changed_future_X[train_index])
    np.testing.assert_array_equal(unchanged_mean, train_mean)
    np.testing.assert_array_equal(unchanged_scale, train_scale)

    independent_data_rng = np.random.default_rng(151)
    reproduced_X = independent_data_rng.normal(size=(900, 8)) * np.array([3., .5, 2., 1., .2, 4., 1., 2.]) + np.arange(8)
    uniform_draws = independent_data_rng.random(900)
    reproduced_logits = 1.2 * (reproduced_X[:, 0] / 3) - 1.5 * ((reproduced_X[:, 1] - 1) / .5) + intercept
    reproduced_y = (uniform_draws < sigmoid(reproduced_logits)).astype(float)
    np.testing.assert_array_equal(raw_X, reproduced_X)
    np.testing.assert_array_equal(raw_y, reproduced_y)
    np.testing.assert_array_equal(order, np.random.default_rng(152).permutation(900))

    audited_runs = []
    for l2, group in runs.items():
        for seed, run in zip(seeds, group):
            minimum_epoch = int(np.argmin(run['history'][:, 1])) + 1
            assert run['epoch'] == minimum_epoch
            recomputed = logistic_objective(X_valid, y_valid, run['w'], run['b'])[1]
            np.testing.assert_allclose(recomputed, run['validation_bce'], rtol=0, atol=1e-15)
            audited_runs.append({'l2': l2, 'seed': seed, 'selected_epoch': minimum_epoch,
                                 'selected_validation_bce': recomputed,
                                 'last_epoch_validation_bce': run['history'][-1, 1]})
    assert selected_l2 == min(mean_validation, key=mean_validation.get)
    np.testing.assert_allclose(seed_nll.std(ddof=1), np.sqrt(np.sum((seed_nll-seed_nll.mean())**2)/2))

    # 호출한 실제 배치의 길이를 계측하고, 즉시 원래 함수를 복원합니다.
    original_objective = globals()['logistic_objective']
    batch_calls = []
    def recording_objective(X, y, w, b, l2=0.):
        batch_calls.append(len(y))
        return original_objective(X, y, w, b, l2)
    globals()['logistic_objective'] = recording_objective
    try:
        train_logistic(X_train, y_train, X_valid, y_valid, epochs=1, seed=1)
    finally:
        globals()['logistic_objective'] = original_objective
    assert batch_calls == [64]*8 + [28, 540, 180]
    last_batch = np.random.default_rng(450).normal(size=(28, 8))
    last_labels = np.array([0., 1.]*14)
    trial_w, trial_b = np.arange(8)/10, .3
    _, _, batch_dw, batch_db = logistic_objective(last_batch, last_labels, trial_w, trial_b, .02)
    scalar_dw = .02 * trial_w.copy()
    scalar_db = 0.
    for row, truth in zip(last_batch, last_labels):
        residual = (float(sigmoid(row @ trial_w + trial_b)) - truth)/28
        scalar_dw += row * residual
        scalar_db += residual
    np.testing.assert_allclose(batch_dw, scalar_dw, rtol=0, atol=1e-15)
    np.testing.assert_allclose(batch_db, scalar_db, rtol=0, atol=1e-15)

    checker_adversaries = {}
    for candidate in [wrong_strict_threshold, wrong_f1_arithmetic_mean,
                      wrong_single_class_auc_zero, wrong_auc_without_tie_credit,
                      wrong_finite_nll_by_clipping, wrong_confusion_transpose]:
        try:
            message = check_binary_metrics(candidate)
            checker_adversaries[candidate.__name__] = {'accepted': True, 'message': message}
        except Exception as error:
            checker_adversaries[candidate.__name__] = {
                'accepted': False, 'exception': type(error).__name__, 'message': str(error)}
    counterexamples = {}
    for name, candidate, y, p, threshold in [
        ('threshold_equality', wrong_strict_threshold, np.array([0, 1]), np.array([.5, .5]), .5),
        ('unequal_precision_recall', wrong_f1_arithmetic_mean, np.array([1, 1, 1, 0]), np.array([.9, .4, .3, .1]), .5),
        ('single_class_auc', wrong_single_class_auc_zero, np.array([0, 0]), np.array([.1, .2]), .5),
    ]:
        counterexamples[name] = {'candidate': candidate.__name__, 'y': y, 'p': p,
                                'threshold': threshold, 'expected': binary_metrics(y, p, threshold),
                                'incorrect': candidate(y, p, threshold)}

    result = {
        'label': label, 'intercept': intercept, 'source_sha256': source_hash,
        'data_seed': 151, 'split_seed': 152, 'training_seeds': seeds,
        'shapes': {name: list(value.shape) for name, value in
                   [('raw_X', raw_X), ('raw_y', raw_y), ('X_train', X_train),
                    ('X_valid', X_valid), ('X_test', X_test), ('train_mean', train_mean),
                    ('train_scale', train_scale), ('w', selected_runs[0]['w']),
                    ('b', np.asarray(selected_runs[0]['b'])), ('ensemble_probability', ensemble_probability)]},
        'array_hashes': {'raw_X': _array_hash(raw_X), 'raw_y': _array_hash(raw_y),
                         'split_order': _array_hash(order), 'label_uniform_draws': _array_hash(uniform_draws)},
        'split_counts': [len(indices) for indices in split_groups],
        'positive_counts': [int(raw_y[indices].sum()) for indices in split_groups],
        'positive_prevalence': [float(raw_y[indices].mean()) for indices in split_groups],
        'split_disjoint_complete': True,
        'preprocessing_train_only': True,
        'train_mean': train_mean, 'train_scale': train_scale,
        'standardized_train_mean_max_abs': float(np.max(np.abs(X_train.mean(axis=0)))),
        'standardized_train_std_max_error': float(np.max(np.abs(X_train.std(axis=0)-1))),
        'group_leakage_example': {'row_accuracy': row_accuracy, 'group_accuracy': group_accuracy,
                                  'row_split_shared_groups': row_overlap},
        'gradient_max_relative_error': max(check_errors),
        'tiny_bce': tiny['validation_bce'],
        'tiny_accuracy': float(np.mean((tiny_prediction >= .5) == y_tiny)),
        'batch_observation_counts_then_epoch_evaluations': batch_calls,
        'last_batch_size_28_scalar_gradient_max_abs_error': float(np.max(np.abs(batch_dw-scalar_dw))),
        'mean_validation_bce_by_l2': mean_validation,
        'all_runs': audited_runs,
        'selected_l2': selected_l2,
        'selected_epochs': [run['epoch'] for run in selected_runs],
        'selected_threshold': selected_threshold,
        'best_validation_f1': float(np.max(threshold_f1)),
        'threshold_maximizers': thresholds[threshold_f1 == np.max(threshold_f1)],
        'own_checker_message': checker_message,
        'own_validation_and_test_metrics_match': True,
        'ensemble_test': own_test,
        'seed_test_metrics': test_results,
        'test_seed_nll_mean': float(seed_nll.mean()),
        'test_seed_nll_std_ddof_1': float(seed_nll.std(ddof=1)),
        'seed_interpretation': '고정된 데이터와 분할에서 초기화·배치 순서에 따른 학습 및 checkpoint 선택의 변동입니다. 새 데이터 표본의 불확실성 추정이 아닙니다.',
        'calibration': {'bin_mean_probability': bin_probabilities, 'bin_positive_frequency': bin_frequencies,
                         'counts': bin_counts, 'ece': ece},
        'validation_original': ordinary_metrics,
        'validation_sharpened_logits': sharp_metrics,
        'validation_permuted_feature_nll': permuted_nll,
        'checker_adversaries': checker_adversaries,
        'counterexamples': counterexamples,
        'notebook_experiment_record': experiment_record,
    }
    target = Path(target_dir) / f'{label}_metrics.json'
    target.write_text(json.dumps(_jsonable(result), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print('독립 구현 및 실험 검산:', checker_message)
    print('REHEARSAL', label, json.dumps(_jsonable({
        'positive_prevalence': result['positive_prevalence'],
        'selected_l2': selected_l2, 'selected_epochs': result['selected_epochs'],
        'selected_threshold': selected_threshold, 'ensemble_test': own_test,
        'accepted_incorrect_candidates': [name for name, value in checker_adversaries.items() if value['accepted']],
    }), ensure_ascii=False))
    return result
