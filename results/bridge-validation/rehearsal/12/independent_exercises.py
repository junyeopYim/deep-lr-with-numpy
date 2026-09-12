
# Independent KL from E_q[log q] - E_q[log p], evaluated per observation and coordinate.
# q variance is exp(q_logvar); p is fixed and shared across observations.
learner_calls=0
def my_diagonal_kl(q_mu,q_logvar,p_mu,p_logvar):
    global learner_calls
    learner_calls+=1
    N,D=q_mu.shape
    dmu=np.zeros((N,D))
    dlogvar=np.zeros((N,D))
    summed_kl=0.0
    for n in range(N):
        for j in range(D):
            q_variance=np.exp(q_logvar[n,j])
            p_variance=np.exp(p_logvar[j])
            difference=q_mu[n,j]-p_mu[j]
            expected_log_q=-0.5*(np.log(2*np.pi)+q_logvar[n,j]+1)
            expected_log_p=-0.5*(np.log(2*np.pi)+p_logvar[j]+(q_variance+difference**2)/p_variance)
            summed_kl+=expected_log_q-expected_log_p
            dmu[n,j]=difference/p_variance/N
            dlogvar[n,j]=0.5*(q_variance/p_variance-1)/N
    return summed_kl/N,dmu,dlogvar
learner_result=check_diagonal_kl(my_diagonal_kl)
initial_checker_calls=learner_calls
assert initial_checker_calls==4

# Start with exact standard-normal cases, then vary means, variances and dimensions.
mean_hand=my_diagonal_kl(np.array([[1.,2.]]),np.zeros((1,2)),np.zeros(2),np.zeros(2))
np.testing.assert_allclose(mean_hand[0],2.5)
np.testing.assert_allclose(mean_hand[1],[[1.,2.]])
np.testing.assert_allclose(mean_hand[2],0)
variance_hand=my_diagonal_kl(np.zeros((1,1)),np.log([[4.]]),np.zeros(1),np.zeros(1))
np.testing.assert_allclose(variance_hand[0],0.5*(3-np.log(4)))
np.testing.assert_allclose(variance_hand[1],0)
np.testing.assert_allclose(variance_hand[2],[[1.5]])

variation_rng=np.random.default_rng(130)
variation_qm=variation_rng.normal(size=(3,4))
variation_ql=np.log(np.array([[0.2,0.5,2.,4.],[1.3,0.3,1.2,3.],[0.7,2.2,0.4,1.]]))
variation_pm=np.array([-0.3,0.2,1.,-0.5])
variation_pl=np.log([0.4,1.7,2.5,0.8])
variation_loss,variation_dm,variation_dl=my_diagonal_kl(variation_qm,variation_ql,variation_pm,variation_pl)
variation_numeric_mu=numerical_gradient(lambda:my_diagonal_kl(variation_qm,variation_ql,variation_pm,variation_pl)[0],variation_qm)
variation_numeric_l=numerical_gradient(lambda:my_diagonal_kl(variation_qm,variation_ql,variation_pm,variation_pl)[0],variation_ql)
variation_gradient_errors={'mu':float(rel_error(variation_dm,variation_numeric_mu)),
                           'logvar':float(rel_error(variation_dl,variation_numeric_l))}
assert max(variation_gradient_errors.values())<1e-7
tripled_loss,tripled_dm,tripled_dl=my_diagonal_kl(np.repeat(variation_qm,3,axis=0),np.repeat(variation_ql,3,axis=0),variation_pm,variation_pl)
np.testing.assert_allclose(tripled_loss,variation_loss)
np.testing.assert_allclose(tripled_dm,np.repeat(variation_dm,3,axis=0)/3)
np.testing.assert_allclose(tripled_dl,np.repeat(variation_dl,3,axis=0)/3)
identity_loss,identity_dm,identity_dl=my_diagonal_kl(variation_pm[None],variation_pl[None],variation_pm,variation_pl)
np.testing.assert_allclose(identity_loss,0,atol=1e-14)
np.testing.assert_allclose(identity_dm,0)
np.testing.assert_allclose(identity_dl,0)

# Rebuild the requested mean-shift curves with the learner function.
learner_curves={}
curvature_checks={}
for p_variance in [0.5,1.,2.]:
    values=[my_diagonal_kl(np.array([[m]]),np.zeros((1,1)),np.zeros(1),np.log([p_variance]))[0] for m in means]
    learner_curves[str(p_variance)]=values
    np.testing.assert_allclose(values,kl_by_variance[f'p 분산 {p_variance}'])
    at_zero=my_diagonal_kl(np.array([[0.]]),np.zeros((1,1)),np.zeros(1),np.log([p_variance]))[0]
    at_one=my_diagonal_kl(np.array([[1.]]),np.zeros((1,1)),np.zeros(1),np.log([p_variance]))[0]
    np.testing.assert_allclose(at_one-at_zero,1/(2*p_variance))
    curvature_checks[str(p_variance)]={'mean_zero_kl':float(at_zero),'mean_one_kl':float(at_one),'mean_shift_cost':float(at_one-at_zero)}

# Deliberate mistakes: correct shapes alone must not be enough.
def probe_checker(candidate):
    try:
        result=check_diagonal_kl(candidate)
    except Exception as exc:
        return {'rejected':True,'exception_type':type(exc).__name__,'exception':str(exc)}
    return {'rejected':False,'result':result}
def wrong_reverse(qm,ql,pm,pl):
    N=len(qm)
    difference=qm-pm
    qvar,pvar=np.exp(ql),np.exp(pl)
    value=0.5*np.sum(ql-pl+(pvar+difference**2)/qvar-1)/N
    dm=difference/qvar/N
    dl=0.5*(1-(pvar+difference**2)/qvar)/N
    return value,dm,dl
def wrong_coordinate_mean(qm,ql,pm,pl):
    value,dm,dl=my_diagonal_kl(qm,ql,pm,pl)
    return value/qm.shape[1],dm/qm.shape[1],dl/qm.shape[1]
def wrong_no_batch_mean(qm,ql,pm,pl):
    value,dm,dl=my_diagonal_kl(qm,ql,pm,pl)
    return value*len(qm),dm*len(qm),dl*len(qm)
def wrong_log_std(qm,ql,pm,pl):
    value,dm,dl=my_diagonal_kl(qm,2*ql,pm,2*pl)
    return value,dm,2*dl
def wrong_logvar_gradient(qm,ql,pm,pl):
    value,dm,dl=my_diagonal_kl(qm,ql,pm,pl)
    return value,dm,2*dl
def wrong_shape(qm,ql,pm,pl):
    value,dm,dl=my_diagonal_kl(qm,ql,pm,pl)
    return value,dm[None],dl
checker_probes={
    'reverse_kl':probe_checker(wrong_reverse),
    'coordinate_mean':probe_checker(wrong_coordinate_mean),
    'missing_batch_mean':probe_checker(wrong_no_batch_mean),
    'logvar_as_logstd':probe_checker(wrong_log_std),
    'logvar_gradient_factor':probe_checker(wrong_logvar_gradient),
    'wrong_shape':probe_checker(wrong_shape),
    'empty_return':probe_checker(lambda qm,ql,pm,pl:())}
assert all(value['rejected'] for value in checker_probes.values())

# Shared Gaussian parameters differ from independent per-observation q parameters.
shared_mu=np.array([0.1,-0.2])
shared_logvar=np.log([0.7,2.])
shared_original=diagonal_nll(X_fit[:7],shared_mu,shared_logvar)
shared_repeated=diagonal_nll(np.repeat(X_fit[:7],3,axis=0),shared_mu,shared_logvar)
for original,repeated in zip(shared_original,shared_repeated):
    np.testing.assert_allclose(original,repeated)

# Direct density-ratio Monte Carlo, sampled from q, with its sampling standard error.
mc_rng=np.random.default_rng(131)
mc_qm=variation_qm[0]
mc_ql=variation_ql[0]
mc_z=mc_rng.normal(size=(50000,4))*np.exp(0.5*mc_ql)+mc_qm
log_ratio=gaussian_logpdf(mc_z,mc_qm,np.diag(np.exp(mc_ql)))-gaussian_logpdf(mc_z,variation_pm,np.diag(np.exp(variation_pl)))
independent_exact=my_diagonal_kl(mc_qm[None],mc_ql[None],variation_pm,variation_pl)[0]
independent_estimate=float(log_ratio.mean())
independent_standard_error=float(log_ratio.std(ddof=1)/np.sqrt(len(log_ratio)))
assert abs(independent_estimate-independent_exact)<5*independent_standard_error

# Confirm negative differential entropy and the MAP sum/mean scaling.
negative_entropy=0.5*(2*(1+np.log(2*np.pi))+np.linalg.slogdet(0.01*np.eye(2))[1])
assert negative_entropy<0
probe_theta=np.linspace(-1,3,13)
summed_objective=np.array([np.sum((observed-t)**2)/(2*sigma2)+t*t/(2*tau2) for t in probe_theta])
mean_objective=np.array([np.mean((observed-t)**2)/(2*sigma2)+t*t/(2*len(observed)*tau2) for t in probe_theta])
np.testing.assert_allclose(summed_objective/len(observed),mean_objective)
map_after_replication=np.tile(observed,2).sum()/(2*len(observed)+sigma2/tau2)
np.testing.assert_allclose(map_after_replication,1.5)

# Reparameterization for per-observation parameters and for shared (D,) parameters.
reparam_mu_error=rel_error(upstream,numerical_gradient(lambda:np.sum((q_mu+np.exp(0.5*q_logvar)*fixed_noise)*upstream),q_mu))
assert reparam_mu_error<1e-7
shared_noise=mc_rng.normal(size=(3,2))
shared_G=mc_rng.normal(size=(3,2))
shared_reparam_dmu=shared_G.sum(axis=0)
shared_reparam_dl=(0.5*shared_G*shared_noise*np.exp(0.5*shared_logvar)).sum(axis=0)
shared_objective=lambda:float(np.sum((shared_mu+np.exp(0.5*shared_logvar)*shared_noise)*shared_G))
shared_reparam_errors={
    'mu':float(rel_error(shared_reparam_dmu,numerical_gradient(shared_objective,shared_mu))),
    'logvar':float(rel_error(shared_reparam_dl,numerical_gradient(shared_objective,shared_logvar)))}
assert max(shared_reparam_errors.values())<1e-7

rehearsal_metrics={
    'empirical_mean':samples.mean(axis=0).tolist(),'target_mean':mu.tolist(),
    'empirical_covariance':empirical_covariance.tolist(),'target_covariance':Sigma.tolist(),
    'logpdf_at_mean':float(at_mean[0]),
    'entropy_discrete':float(entropy(p)),'cross_entropy_discrete':float(cross_entropy(p,q)),
    'kl_discrete_pq':float(categorical_kl(p,q)),'kl_discrete_qp':float(categorical_kl(q,p)),
    'gaussian_entropy_exact':float(gaus_entropy),'gaussian_entropy_mc':float(monte_carlo_entropy),
    'negative_differential_entropy_example':float(negative_entropy),
    'nll_initial':float(history[0]),'nll_final':float(final_nll),'fit_steps':1200,
    'nll_gradient_max_error':float(max(grad_errors)),
    'learned_mean':mu_fit.tolist(),'learned_variances':np.exp(logvar_fit).tolist(),
    'mean_fit_max_error':float(np.max(np.abs(mu_fit-X_fit.mean(axis=0)))),
    'variance_fit_max_error':float(np.max(np.abs(np.exp(logvar_fit)-X_fit.var(axis=0)))),
    'full_holdout_nll':float(full_holdout_nll),'diagonal_holdout_nll':float(diagonal_holdout_nll),
    'full_fitted_covariance':full_covariance.tolist(),
    'mle':float(theta_mle),'map':float(theta_map),'map_after_data_replication':float(map_after_replication),
    'kl_exact':float(kl),'kl_mc':float(np.mean(mc_estimates)),
    'kl_gradient_errors':[float(value) for value in kl_grad_errors],
    'reparam_logvar_error':float(reparam_error),'reparam_mu_error':float(reparam_mu_error),
    'shared_reparameterization_errors':shared_reparam_errors,
    'learner_checker_result':learner_result,'initial_checker_candidate_calls':initial_checker_calls,
    'standard_normal_mean_hand':[float(mean_hand[0]),mean_hand[1].tolist(),mean_hand[2].tolist()],
    'variance_hand':[float(variance_hand[0]),variance_hand[1].tolist(),variance_hand[2].tolist()],
    'variation_shape':[3,4],'variation_kl':float(variation_loss),'variation_gradient_errors':variation_gradient_errors,
    'tripled_batch_shape':[9,4],'tripled_kl':float(tripled_loss),'tripled_gradient_scaling_verified':True,
    'identity_kl':float(identity_loss),'learner_curvature_checks':curvature_checks,
    'checker_probes':checker_probes,'shared_nll_replication_verified':True,
    'independent_mc':{'q_sample_count':50000,'exact_kl':float(independent_exact),'estimated_kl':independent_estimate,
                     'standard_error':independent_standard_error,'standardized_error':float((independent_estimate-independent_exact)/independent_standard_error)}}
import json
print('REHEARSAL_JSON '+json.dumps(rehearsal_metrics))
