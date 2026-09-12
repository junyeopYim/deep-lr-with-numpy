import json

# 독립 풀이: 각 관측과 좌표의 Gaussian KL을 수식 그대로 더합니다.
# 원본 diagonal_kl을 호출하지 않으며, 분산을 먼저 복원한 반복문입니다.
def my_diagonal_kl(q_mu, q_logvar, p_mu, p_logvar):
    N, D = q_mu.shape
    loss = 0.0
    dmu = np.zeros((N, D), dtype=float)
    dlogvar = np.zeros((N, D), dtype=float)
    for n in range(N):
        for j in range(D):
            variance_q = np.exp(q_logvar[n, j])
            variance_p = np.exp(p_logvar[j])
            delta = q_mu[n, j] - p_mu[j]
            loss += (np.log(variance_p / variance_q)
                     + (variance_q + delta * delta) / variance_p - 1) / (2 * N)
            dmu[n, j] = delta / (N * variance_p)
            dlogvar[n, j] = (variance_q / variance_p - 1) / (2 * N)
    return loss, dmu, dlogvar

checker_result = check_diagonal_kl(my_diagonal_kl)
print('직접 쓴 KL:', checker_result)

# p의 분산이 1일 때 손계산: 0.5*(1+2-1-ln2 + .25+.5-1-ln.5) = .875
hand_qmu = np.array([[1., -.5]])
hand_qvar = np.array([[2., .5]])
hand_loss, hand_dmu, hand_dlogvar = my_diagonal_kl(
    hand_qmu, np.log(hand_qvar), np.zeros(2), np.zeros(2))
np.testing.assert_allclose(hand_loss, .875, atol=1e-14)
np.testing.assert_allclose(hand_dmu, [[1., -.5]])
np.testing.assert_allclose(hand_dlogvar, [[.5, -.25]])

# 새 입력: N=3, D=4, 비표준 p. 두 미분을 독립 풀이의 손실로도 차분합니다.
new_qmu = np.array([[.2, -.5, 1.2, .7], [1.1, .4, -.2, -.6], [-.1, .8, .9, 1.3]])
new_qlogvar = np.log(np.array([[.5, 2., .8, 1.5], [1.2, .7, 3., .4], [.9, 1.7, 1.1, 2.2]]))
new_pmu = np.array([-.2, .4, 1., 0.])
new_plogvar = np.log(np.array([.25, 1., 2., 4.]))
new_loss, new_dmu, new_dlogvar = my_diagonal_kl(new_qmu, new_qlogvar, new_pmu, new_plogvar)
new_grad_errors = [
    rel_error(new_dmu, numerical_gradient(lambda: my_diagonal_kl(new_qmu, new_qlogvar, new_pmu, new_plogvar)[0], new_qmu)),
    rel_error(new_dlogvar, numerical_gradient(lambda: my_diagonal_kl(new_qmu, new_qlogvar, new_pmu, new_plogvar)[0], new_qlogvar)),
]
assert max(new_grad_errors) < 1e-7
repeat_loss, repeat_dmu, repeat_dl = my_diagonal_kl(
    np.repeat(new_qmu, 2, axis=0), np.repeat(new_qlogvar, 2, axis=0), new_pmu, new_plogvar)
np.testing.assert_allclose(repeat_loss, new_loss)
np.testing.assert_allclose(repeat_dmu, np.repeat(new_dmu, 2, axis=0)/2)
np.testing.assert_allclose(repeat_dl, np.repeat(new_dlogvar, 2, axis=0)/2)

# 의도적 오답: KL 방향, 기준 p, 좌표 합, 관측 평균 미분 규약을 각각 위반합니다.
def wrong_reverse_kl(qm, ql, pm, pl):
    N = len(qm)
    delta = qm - pm
    ratio_sum = (np.exp(pl) + delta**2) / np.exp(ql)
    loss = .5 * np.sum(ql - pl + ratio_sum - 1) / N
    return loss, delta * np.exp(-ql) / N, .5 * (1 - ratio_sum) / N


def wrong_standard_normal(qm, ql, pm, pl):
    return (.5 * np.sum(qm**2 + np.exp(ql) - 1 - ql) / len(qm),
            qm / len(qm), .5 * (np.exp(ql)-1) / len(qm))


def wrong_coordinate_mean(qm, ql, pm, pl):
    value, dm, dl = my_diagonal_kl(qm, ql, pm, pl)
    return value / qm.shape[1], dm / qm.shape[1], dl / qm.shape[1]


def wrong_missing_gradient_batch_mean(qm, ql, pm, pl):
    value, dm, dl = my_diagonal_kl(qm, ql, pm, pl)
    return value, dm * len(qm), dl * len(qm)

negative_results = {}
for name, candidate in [
    ('reverse_kl', wrong_reverse_kl),
    ('ignores_nonstandard_p', wrong_standard_normal),
    ('mean_over_coordinates', wrong_coordinate_mean),
    ('missing_gradient_batch_mean', wrong_missing_gradient_batch_mean),
]:
    try:
        outcome = check_diagonal_kl(candidate)
    except AssertionError as exc:
        negative_results[name] = {'rejected': True, 'error_type': type(exc).__name__, 'message': str(exc).strip()}
    else:
        negative_results[name] = {'rejected': False, 'outcome': outcome}
assert all(v['rejected'] for v in negative_results.values())

# 7절 변형: q의 두 번째 평균만 움직이고, 해당 축 p의 분산을 달리합니다.
# N=3으로 하고 첫 번째 축의 KL은 0, q의 두 번째 분산은 1로 고정합니다.
my_shifts = np.linspace(-2, 2, 101)
my_curves = {}
shift_costs = {}
for pv in [.25, 1., 4.]:
    pm = np.array([.4, -.7])
    pl = np.log(np.array([2., pv]))
    ql = np.tile(np.log([2., 1.]), (3, 1))
    values = []
    for shift in my_shifts:
        qm = np.tile(pm + [0., shift], (3, 1))
        values.append(my_diagonal_kl(qm, ql, pm, pl)[0])
    values = np.array(values)
    base = .5 * (np.log(pv) + 1/pv - 1)
    np.testing.assert_allclose(values, base + my_shifts**2/(2*pv), atol=1e-14)
    my_curves[f'p의 둘째 축 분산 {pv}'] = values
    shift_costs[str(pv)] = {'at_zero': float(values[50]), 'at_one': float(values[75]),
                            'increase_for_unit_shift': float(values[75]-values[50]),
                            'expected_increase': 1/(2*pv)}
plot_curves(my_shifts, my_curves, title='직접 변형: 둘째 평균 이동과 기준 분산',
            xlabel='q의 둘째 평균 − p의 둘째 평균', ylabel='관측 평균 KL, nat')

# 행 샘플의 L.T 계약을 작은 네 점으로 재현합니다.
epsilon_four = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
row_samples = epsilon_four @ L_hand.T + mu
row_center = row_samples - mu
hand_covariance = row_center.T @ row_center / 4
np.testing.assert_allclose(hand_covariance, Sigma)
wrong_rows = epsilon_four @ L_hand + mu
wrong_center = wrong_rows - mu
wrong_covariance = wrong_center.T @ wrong_center / 4
assert not np.allclose(wrong_covariance, Sigma)

# MAP의 합/평균 규약: J/N이면 데이터 항과 사전 항 모두 N으로 나눕니다.
map_average_stationary = np.mean(theta_map-observed)/sigma2 + theta_map/(len(observed)*tau2)
np.testing.assert_allclose(map_average_stationary, 0, atol=1e-14)
map_more_data = np.repeat(observed, 4).sum() / (len(observed)*4 + sigma2/tau2)
np.testing.assert_allclose(map_more_data, 12/7)

# 고정 잡음의 z 변화를 수식에서 다시 미분합니다. 기대값 미분과 혼동하지 않습니다.
reparam_mu = np.array([[.4, -.8], [1.1, .2]])
reparam_logvar = np.log(np.array([[.25, 4.], [2., .5]]))
noise_hand = np.array([[1., -2.], [.5, 0.]])
upstream_hand = np.array([[2., -1.], [.3, .7]])
sample_hand = reparam_mu + np.exp(reparam_logvar / 2) * noise_hand
sample_mu_grad = upstream_hand.copy()
sample_logvar_grad = upstream_hand * noise_hand * np.exp(reparam_logvar / 2) / 2
sample_gradient_errors = [
    rel_error(sample_mu_grad, numerical_gradient(lambda: np.sum((reparam_mu + np.exp(reparam_logvar/2)*noise_hand)*upstream_hand), reparam_mu)),
    rel_error(sample_logvar_grad, numerical_gradient(lambda: np.sum((reparam_mu + np.exp(reparam_logvar/2)*noise_hand)*upstream_hand), reparam_logvar)),
]
assert max(sample_gradient_errors) < 1e-7
np.testing.assert_allclose(sample_logvar_grad[0], [.5, 2.])

# 본문의 공유 (D,) 파라미터 계약: broadcasting 역전파는 샘플 축을 합합니다.
shared_mu = np.array([.4, -.8])
shared_logvar = np.log(np.array([.25, 4.]))
shared_dmu = upstream_hand.sum(axis=0)
shared_dl = (.5 * upstream_hand * noise_hand * np.exp(.5*shared_logvar)).sum(axis=0)
shared_gradient_errors = [
    rel_error(shared_dmu, numerical_gradient(lambda: np.sum((shared_mu + np.exp(.5*shared_logvar)*noise_hand)*upstream_hand), shared_mu)),
    rel_error(shared_dl, numerical_gradient(lambda: np.sum((shared_mu + np.exp(.5*shared_logvar)*noise_hand)*upstream_hand), shared_logvar)),
]
assert max(shared_gradient_errors) < 1e-7

rehearsal_metrics = {
    'checker_result': checker_result,
    'hand_standard_p': {'loss': hand_loss, 'dmu': hand_dmu.tolist(), 'dlogvar': hand_dlogvar.tolist()},
    'new_input': {'N': 3, 'D': 4, 'loss': new_loss, 'gradient_relative_errors': new_grad_errors,
                  'repeat_loss_difference': abs(repeat_loss-new_loss),
                  'gradient_shapes': [list(new_dmu.shape), list(new_dlogvar.shape)]},
    'negative_candidates': negative_results,
    'mean_shift_variant': shift_costs,
    'row_sampling_hand': {'correct_covariance': hand_covariance.tolist(), 'without_transpose_covariance': wrong_covariance.tolist()},
    'original_notebook': {
        'sample_count': len(samples), 'empirical_mean': samples.mean(axis=0).tolist(),
        'empirical_covariance': empirical_covariance.tolist(),
        'logpdf_at_mean': float(at_mean[0]),
        'categorical_entropy': float(entropy(p)), 'categorical_cross_entropy': float(cross_entropy(p,q)),
        'categorical_kl': float(categorical_kl(p,q)), 'categorical_reverse_kl': float(categorical_kl(q,p)),
        'gaussian_entropy_exact': float(gaus_entropy), 'gaussian_entropy_monte_carlo': float(monte_carlo_entropy),
        'nll_initial': float(history[0]), 'nll_final': float(final_nll),
        'nll_gradient_errors': grad_errors,
        'mean_fit_max_abs_error': float(np.max(np.abs(mu_fit-X_fit.mean(axis=0)))),
        'variance_fit_max_abs_error': float(np.max(np.abs(np.exp(logvar_fit)-X_fit.var(axis=0)))),
        'holdout_nll_full': float(full_holdout_nll), 'holdout_nll_diagonal': float(diagonal_holdout_nll),
        'theta_mle': float(theta_mle), 'theta_map': float(theta_map),
        'kl_exact': float(kl), 'kl_monte_carlo': float(np.mean(mc_estimates)),
        'kl_gradient_errors': kl_grad_errors, 'fixed_noise_gradient_error': float(reparam_error),
    },
    'map_average_stationary': float(map_average_stationary),
    'map_repeated_observations': float(map_more_data),
    'independent_reparameterization': {'z': sample_hand.tolist(), 'dmu': sample_mu_grad.tolist(),
                                      'dlogvar': sample_logvar_grad.tolist(),
                                      'gradient_relative_errors': sample_gradient_errors},
    'shared_reparameterization': {'dmu': shared_dmu.tolist(), 'dlogvar': shared_dl.tolist(),
                                 'gradient_relative_errors': shared_gradient_errors},
}
print('REHEARSAL_METRICS=' + json.dumps(rehearsal_metrics, ensure_ascii=False))
