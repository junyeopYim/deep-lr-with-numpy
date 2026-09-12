"""Notebook globals are provided by the freshly executed source snapshot.

The exercise solution uses scalar loops and does not call either source loss.
All other code below is an independent rehearsal probe, not repository code.
"""
import itertools
import json

candidate_call_count = 0


def my_regression_loss(prediction, target, kind='huber', delta=.5, weights=None):
    global candidate_call_count
    candidate_call_count += 1
    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    if prediction.ndim != 2 or prediction.shape != target.shape:
        raise ValueError('예측과 목표는 같은 (N,K) 배열이어야 합니다.')
    N, K = prediction.shape
    weights = np.ones(N) if weights is None else np.asarray(weights, dtype=float)
    if (weights.shape != (N,) or not np.all(np.isfinite(weights))
            or np.any(weights < 0) or weights.sum() <= 0):
        raise ValueError('유한한 비음수 관측 가중치와 양수 합이 필요합니다.')
    if kind not in ('mse', 'mae', 'huber') or (kind == 'huber' and delta <= 0):
        raise ValueError('손실 종류 또는 delta를 확인하세요.')
    divisor = K * sum(float(a) for a in weights)
    value = 0.0
    gradient = np.zeros((N, K), dtype=float)
    for n in range(N):
        for k in range(K):
            residual = float(prediction[n, k] - target[n, k])
            sign = 1.0 if residual > 0 else (-1.0 if residual < 0 else 0.0)
            if kind == 'mse':
                element, derivative = residual * residual / 2.0, residual
            elif kind == 'mae':
                element, derivative = abs(residual), sign
            elif abs(residual) <= delta:
                element, derivative = residual * residual / 2.0, residual
            else:
                element = delta * abs(residual) - delta * delta / 2.0
                derivative = delta * sign
            value += weights[n] * element / divisor
            gradient[n, k] = weights[n] * derivative / divisor
    return float(value), gradient


metrics = {}
before = candidate_call_count
metrics['candidate_checker'] = {
    'result': check_regression_loss(my_regression_loss),
    'calls': candidate_call_count - before,
}
metrics['delta_fits'] = []
for rehearsal_delta in [.1, .3, 1.]:
    before = candidate_call_count
    rehearsal_W, rehearsal_b, rehearsal_history = fit_linear(
        X_train, Y_train, 'huber', delta=rehearsal_delta,
        loss_fn=my_regression_loss)
    rehearsal_rmse = float(np.sqrt(np.mean(
        (X_clean @ rehearsal_W + rehearsal_b - Y_clean)**2)))
    np.testing.assert_allclose(rehearsal_rmse, delta_results[rehearsal_delta], atol=1e-12)
    metrics['delta_fits'].append({
        'delta': rehearsal_delta, 'W': rehearsal_W.tolist(),
        'b': rehearsal_b.tolist(), 'clean_rmse': rehearsal_rmse,
        'initial_objective': float(rehearsal_history[0]),
        'final_objective': float(rehearsal_history[-1]),
        'updates': len(rehearsal_history)-1,
        'candidate_calls': candidate_call_count - before,
    })

# Hand values use all three signs, a zero weight, and two output coordinates.
hand_prediction = np.array([[2., -1.], [.25, -.25], [8., -7.]])
hand_target = np.zeros_like(hand_prediction)
hand_weights = np.array([1., 3., 0.])
hand_mse = (1*(2+.5) + 3*(.03125+.03125))/(2*4)
hand_mae = (1*(2+1) + 3*(.25+.25))/(2*4)
hand_huber = (1*(.875+.375) + 3*(.03125+.03125))/(2*4)
metrics['hand_weighted_losses'] = {}
metrics['independent_gradient_errors'] = {}
for rehearsal_kind, hand_expected in [('mse', hand_mse), ('mae', hand_mae), ('huber', hand_huber)]:
    hand_value, hand_gradient = my_regression_loss(
        hand_prediction, hand_target, rehearsal_kind, .5, hand_weights)
    np.testing.assert_allclose(hand_value, hand_expected, atol=1e-14)
    assert np.all(hand_gradient[2] == 0)
    numeric = numerical_gradient(lambda: my_regression_loss(
        hand_prediction, hand_target, rehearsal_kind, .5, hand_weights)[0], hand_prediction)
    error = rel_error(hand_gradient, numeric)
    assert error < 1e-7
    metrics['independent_gradient_errors'][rehearsal_kind] = error
    scaled_value, scaled_gradient = my_regression_loss(
        hand_prediction, hand_target, rehearsal_kind, .5, hand_weights*7)
    np.testing.assert_allclose(scaled_value, hand_value, atol=1e-14)
    np.testing.assert_allclose(scaled_gradient, hand_gradient, atol=1e-14)
    duplicate_value, duplicate_gradient = my_regression_loss(
        np.repeat(hand_prediction, 2, axis=1), np.repeat(hand_target, 2, axis=1),
        rehearsal_kind, .5, hand_weights)
    np.testing.assert_allclose(duplicate_value, hand_value, atol=1e-14)
    np.testing.assert_allclose(duplicate_gradient, np.repeat(hand_gradient, 2, axis=1)/2, atol=1e-14)
    metrics['hand_weighted_losses'][rehearsal_kind] = {
        'value': hand_value, 'gradient': hand_gradient.tolist(),
        'zero_weight': True, 'weight_rescaling': True, 'output_duplication': True,
    }

# A derivative at a MAE kink is a chosen subgradient; Huber's first derivative joins.
for boundary in [-.5, .5]:
    loss_at, grad_at = my_regression_loss(np.array([[boundary]]), np.zeros((1, 1)), 'huber', .5)
    np.testing.assert_allclose(loss_at, .125)
    np.testing.assert_allclose(grad_at.item(), boundary)
np.testing.assert_allclose(my_regression_loss(np.zeros((1, 1)), np.zeros((1, 1)), 'mae')[1], 0.)
metrics['boundary_and_subgradient'] = True


def wrong_candidate(mode):
    def candidate(prediction, target, kind='huber', delta=.5, weights=None):
        if mode == 'incomplete_return':
            return None
        if mode == 'ignore_weights':
            return my_regression_loss(prediction, target, kind, delta, None)
        if mode == 'ignore_delta':
            return my_regression_loss(prediction, target, kind, .5, weights)
        value, grad = my_regression_loss(prediction, target, kind, delta, weights)
        N, K = prediction.shape
        if mode == 'mse_without_half' and kind == 'mse':
            return 2*value, 2*grad
        if mode == 'mae_wrong_sign' and kind == 'mae':
            return value, -grad
        if mode == 'huber_missing_offset' and kind == 'huber':
            a = np.ones(N) if weights is None else np.asarray(weights)
            tail = np.abs(prediction-target) > delta
            return value + .5*delta**2*np.sum(a[:, None]*tail)/(K*a.sum()), grad
        if mode == 'huber_missing_delta_gradient' and kind == 'huber':
            grad[np.abs(prediction-target) > delta] /= delta
        if mode == 'missing_output_average':
            return K*value, K*grad
        if mode == 'normalise_by_N':
            factor = 1. if weights is None else np.sum(weights)/N
            return value*factor, grad*factor
        if mode == 'missing_batch_average':
            return N*value, N*grad
        if mode == 'gradient_wrong_shape':
            return value, grad.sum(axis=1)
        if mode == 'all_zero':
            return 0., np.zeros_like(prediction)
        return value, grad
    return candidate


metrics['wrong_candidate_results'] = {}
for mode in ['mse_without_half', 'mae_wrong_sign', 'huber_missing_offset',
             'huber_missing_delta_gradient', 'ignore_weights', 'ignore_delta',
             'missing_output_average', 'normalise_by_N', 'missing_batch_average',
             'gradient_wrong_shape', 'all_zero', 'incomplete_return']:
    try:
        message = check_regression_loss(wrong_candidate(mode))
        metrics['wrong_candidate_results'][mode] = {'rejected': False, 'message': str(message)}
    except Exception as exc:
        metrics['wrong_candidate_results'][mode] = {
            'rejected': True, 'type': type(exc).__name__, 'message': str(exc)[:1000]}
assert all(item['rejected'] for item in metrics['wrong_candidate_results'].values())

# L2 and the bias gradient are checked against the actual objective with our loss.
own_objective, own_dW, own_db = linear_objective(
    X_check, Y_check, W_check, b_check, weights=weights_check, l2=.3,
    loss_fn=my_regression_loss)
own_numeric_W = numerical_gradient(lambda: linear_objective(
    X_check, Y_check, W_check, b_check, weights=weights_check, l2=.3,
    loss_fn=my_regression_loss)[0], W_check)
own_numeric_b = numerical_gradient(lambda: linear_objective(
    X_check, Y_check, W_check, b_check, weights=weights_check, l2=.3,
    loss_fn=my_regression_loss)[0], b_check)
_, unregularized_W, unregularized_b = linear_objective(
    X_check, Y_check, W_check, b_check, weights=weights_check,
    loss_fn=my_regression_loss)
np.testing.assert_allclose(own_dW-unregularized_W, .3*W_check, atol=1e-13)
np.testing.assert_allclose(own_db, unregularized_b, atol=1e-13)
duplicated_objective = linear_objective(
    np.repeat(X_check, 2, axis=0), np.repeat(Y_check, 2, axis=0), W_check, b_check,
    weights=np.repeat(weights_check, 2), l2=.3, loss_fn=my_regression_loss)
for actual, expected in zip(duplicated_objective, (own_objective, own_dW, own_db)):
    np.testing.assert_allclose(actual, expected, atol=1e-13)
metrics['l2'] = {
    'gradient_errors': {'W': rel_error(own_dW, own_numeric_W), 'b': rel_error(own_db, own_numeric_b)},
    'batch_duplication_invariant': True, 'bias_unpenalized': True,
    'wrong_divide_l2_by_N_error': rel_error(unregularized_W+.3*W_check/len(X_check), own_numeric_W),
    'wrong_bias_penalty_error': rel_error(own_db+.3*b_check, own_numeric_b),
}
assert max(metrics['l2']['gradient_errors'].values()) < 1e-7
assert metrics['l2']['wrong_divide_l2_by_N_error'] > 1e-4
assert metrics['l2']['wrong_bias_penalty_error'] > 1e-4

# Extend the notebook's ridge example to K=2 without changing its loss convention.
rehearsal_rng = np.random.default_rng(14014)
ridge_X = rehearsal_rng.normal(size=(60, 3))
ridge_Y = ridge_X @ np.array([[1., -.5], [-2., 1.2], [.5, 2.]]) + np.array([.4, -.2])
ridge_lambda = .3
ridge_W, ridge_b, _ = fit_linear(ridge_X, ridge_Y, 'mse', l2=ridge_lambda,
                                steps=3000, loss_fn=my_regression_loss)
ridge_A = np.column_stack([ridge_X, np.ones(len(ridge_X))])
ridge_P = np.diag([1., 1., 1., 0.])
ridge_theta = np.vstack([ridge_W, ridge_b])
ridge_exact_NK = np.linalg.solve(
    ridge_A.T@ridge_A + ridge_Y.size*ridge_lambda*ridge_P, ridge_A.T@ridge_Y)
ridge_exact_N = np.linalg.solve(
    ridge_A.T@ridge_A + len(ridge_Y)*ridge_lambda*ridge_P, ridge_A.T@ridge_Y)
metrics['ridge_multioutput'] = {
    'N': len(ridge_X), 'K': ridge_Y.shape[1], 'lambda': ridge_lambda,
    'max_parameter_difference_NK': float(np.max(np.abs(ridge_theta-ridge_exact_NK))),
    'max_parameter_difference_N': float(np.max(np.abs(ridge_theta-ridge_exact_N))),
    'theta_fit': ridge_theta.tolist(), 'theta_NK': ridge_exact_NK.tolist(),
    'theta_N': ridge_exact_N.tolist(),
}
np.testing.assert_allclose(ridge_theta, ridge_exact_NK, atol=1e-10)
assert metrics['ridge_multioutput']['max_parameter_difference_N'] > .01

# Independent AdamW update oracle based on weighted sums of all past gradients.
def independent_adam_oracle(W, previous_gradients, lr, beta1, beta2, eps, decay):
    t = len(previous_gradients)
    m = sum((1-beta1)*beta1**(t-i-1)*g for i, g in enumerate(previous_gradients))
    v = sum((1-beta2)*beta2**(t-i-1)*g*g for i, g in enumerate(previous_gradients))
    m_hat, v_hat = m/(1-beta1**t), v/(1-beta2**t)
    return (1-lr*decay)*W-lr*m_hat/(np.sqrt(v_hat)+eps), m, v

oracle_W = np.array([1.2, -.7])
oracle_state = (np.zeros(2), np.zeros(2), 0)
oracle_gradients = []
adam_errors, adam_wrong_errors = [], {}
for step_gradient in [np.array([2., 1e-9]), np.array([-1., 2e-9])]:
    oracle_gradients.append(step_gradient)
    oracle_expected, sum_m, sum_v = independent_adam_oracle(
        oracle_W, oracle_gradients, .1, .5, .75, .03, .2)
    oracle_actual, oracle_state = adam_step(
        oracle_W, step_gradient, oracle_state, lr=.1, beta1=.5, beta2=.75,
        eps=.03, weight_decay=.2)
    np.testing.assert_allclose(oracle_actual, oracle_expected, atol=1e-14)
    np.testing.assert_allclose(oracle_state[0], sum_m, atol=1e-14)
    np.testing.assert_allclose(oracle_state[1], sum_v, atol=1e-14)
    adam_errors.append(float(np.max(np.abs(oracle_actual-oracle_expected))))
    t = len(oracle_gradients)
    mh, vh = sum_m/(1-.5**t), sum_v/(1-.75**t)
    wrong_values = {
        'epsilon_inside_root': .98*oracle_W-.1*mh/np.sqrt(vh+.03),
        'missing_bias_correction': .98*oracle_W-.1*sum_m/(np.sqrt(sum_v)+.03),
        'decay_after_adaptive_update': .98*(oracle_W-.1*mh/(np.sqrt(vh)+.03)),
    }
    for key, value in wrong_values.items():
        adam_wrong_errors.setdefault(key, []).append(float(np.max(np.abs(value-oracle_expected))))
    oracle_W = oracle_actual
metrics['adamw'] = {
    'independent_two_step_max_abs_errors': adam_errors,
    'wrong_formula_max_abs_errors': adam_wrong_errors,
    'first_coupled': W_coupled.tolist(), 'first_decoupled': W_decoupled.tolist(),
    'first_coupled_m': coupled_state[0].tolist(),
    'first_decoupled_m': decoupled_state[0].tolist(),
    'two_step_final': oracle_W.tolist(),
}
assert all(max(errors) > 1e-5 for errors in adam_wrong_errors.values())

# Enumerate independent masks for every observation, instead of a shared mask.
independent_mask_expected_loss = 0.
independent_mask_probability = 0.
for rows in itertools.product(range(4), repeat=len(X_two)):
    row_masks = masks[np.array(rows)]
    probability = float(np.prod(mask_probabilities[np.array(rows)]))
    row_prediction = (X_two*row_masks/q_keep) @ w_two + b_two
    independent_mask_expected_loss += probability * .5*np.mean((row_prediction-y_two)**2)
    independent_mask_probability += probability
np.testing.assert_allclose(independent_mask_probability, 1., atol=1e-14)
np.testing.assert_allclose(independent_mask_expected_loss, base_loss+extra_loss, atol=1e-14)
fixed_numeric = numerical_gradient(lambda: np.sum(H*multiplier*G), H)
wrong_dropout_gradient = G*(multiplier != 0)
dropout_bad_error = rel_error(wrong_dropout_gradient, fixed_numeric)
assert dropout_bad_error > .1

standard_X = (X_two-X_two.mean(axis=0))/X_two.std(axis=0)
standard_extra = (1-q_keep)/(2*q_keep)*np.sum(w_two**2*np.mean(standard_X**2, axis=0))
standard_l2 = (1-q_keep)/(2*q_keep)*np.sum(w_two**2)
np.testing.assert_allclose(standard_extra, standard_l2, atol=1e-14)
metrics['dropout'] = {
    'source_fixed_mask_relative_error': dropout_error,
    'source_monte_carlo_draws': 10000,
    'source_monte_carlo_max_abs_error': float(np.max(np.abs(monte_carlo_output-H))),
    'base_loss': float(base_loss), 'extra_loss': float(extra_loss),
    'shared_4_mask_expected_loss': float(exact_expected_loss),
    'independent_64_mask_expected_loss': float(independent_mask_expected_loss),
    'independent_mask_probability_sum': independent_mask_probability,
    'wrong_backward_missing_inverse_q_error': dropout_bad_error,
    'wrong_unscaled_output_mean_max_bias': float(np.max(np.abs(q_keep*H-H))),
    'standardized_feature_extra_loss': float(standard_extra),
    'standardized_feature_l2_form': float(standard_l2),
}

metrics['notebook'] = {
    'weighted_loss_gradient_errors': loss_gradient_errors,
    'linear_gradient_errors': linear_gradient_errors,
    'clean_rmse_by_loss': rmse_by_loss,
    'ridge_norms': [float(value) for value in ridge_norms],
    'delta_results': {str(key): value for key, value in delta_results.items()},
}
print('REHEARSAL_JSON ' + json.dumps(metrics, ensure_ascii=False))
