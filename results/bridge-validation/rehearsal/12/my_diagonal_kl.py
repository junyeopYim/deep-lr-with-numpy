import numpy as np

learner_calls = 0

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
