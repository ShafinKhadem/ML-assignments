# %%
import numpy as np
from scipy.stats import norm


def floatara(x):
    return np.array(x).astype(float)


def split_readline(f, sep='\t'):
    return floatara(f.readline().split(sep))


def normalize(x: np.ndarray):
    x /= np.sum(x)


# %%
data = None

with open('data.txt', 'r') as datafile:
    data = floatara(datafile.read().splitlines())

T = len(data)

# %%
n = 0
P = None
means = None
stddevs = None

with open('parameters.txt', 'r') as parametersfile:
    n = int(parametersfile.readline())
    P = np.zeros(shape=(n, n))
    for i in range(n):
        P[i] = split_readline(parametersfile)
    means = split_readline(parametersfile)
    stddevs = np.sqrt(split_readline(parametersfile))

n, P, means, stddevs


# %%
def stationary(P):
    coeff = P.T.copy()
    np.fill_diagonal(coeff, coeff.diagonal() - 1)
    coeff[-1].fill(1)
    ordinate = np.zeros(len(coeff))
    ordinate[-1] = 1
    ans = np.linalg.solve(coeff, ordinate)
    return ans


# %%
def viterbi(data, P, means, stddevs):

    def E(state, emission):
        return norm.pdf(emission, means[state], stddevs[state])

    T = len(data)
    n = len(means)
    dp = np.zeros((T + 1, n))
    prv = np.zeros((T + 1, n)).astype(int)
    dp[0] = stationary(P)
    for i in range(1, T + 1):
        for j in range(n):
            ep = E(j, data[i - 1])
            options = [dp[i - 1][k] * P[k][j] for k in range(n)]
            prv[i][j] = np.argmax(options)
            dp[i][j] = ep * options[prv[i][j]]
        normalize(dp[i])
    states = [np.argmax(dp[T])]
    for i in range(T, 1, -1):
        states.append(prv[i][states[-1]])
    states.reverse()
    return states


# %%
labels = ['"El Nino"\r\n', '"La Nina"\r\n']


# with open('states_Viterbi_wo_learning.txt', 'r') as of:
#     output = [1 if i == '"La Nina"\n' else 0 for i in of.readlines()]
# print(output == viterbi(data, P, means, stddevs))
def output(states, output_file_path):
    with open(output_file_path, 'w') as of:
        for i in states:
            of.write(labels[i])


output(viterbi(data, P, means, stddevs), 'states_Viterbi_wo_learning.txt')


# %%
def BaumWelch(data, transmat_prior, means_prior, covars_prior, n_iter=10):
    T = len(data)
    n = len(means_prior)
    P = transmat_prior.copy()
    means = means_prior.copy()
    stddevs = np.sqrt(covars_prior)

    def E(state, emission):
        return norm.pdf(emission, means[state], stddevs[state])

    forward = np.zeros((T + 1, n))
    backward = np.zeros((T + 1, n))
    node_blame = np.zeros((T + 1, n))
    edge_blame = np.zeros((T, n, n))
    for i in range(n_iter):
        # E-step
        ep = np.array([[E(j, data[i - 1]) if i else 0 for j in range(n)]
                       for i in range(T + 1)])
        forward[0] = stationary(P)
        for i in range(1, T + 1):
            for j in range(n):
                forward[i][j] = sum(
                    [forward[i - 1][k] * P[k][j] * ep[i][j] for k in range(n)])
            normalize(forward[i])
        backward[T] = np.ones(n)
        for i in range(T, 1, -1):
            for j in range(n):
                backward[i - 1][j] = sum(
                    [backward[i][k] * P[j][k] * ep[i][k] for k in range(n)])
            normalize(backward[i - 1])
        for i in range(1, T + 1):
            for j in range(n):
                node_blame[i][j] = forward[i][j] * backward[i][j]
                for k in range(n):
                    edge_blame[i - 1][k][j] = forward[
                        i - 1][k] * P[k][j] * ep[i][j] * backward[i][j]
            normalize(node_blame[i])
            normalize(edge_blame[i - 1])

        # M-step
        for i in range(n):
            for j in range(n):
                P[i][j] = sum(edge_blame[1:, i, j])
            normalize(P[i])
            w = node_blame[1:, i]
            means[i] = np.average(data, weights=w)
            stddevs[i] = np.sqrt(
                np.average(np.square(data - means[i]), weights=w))
    return P, means, np.square(stddevs), stationary(P)


# %%
params = BaumWelch(data,
                   transmat_prior=P,
                   means_prior=means,
                   covars_prior=np.square(stddevs),
                   n_iter=10)

# %%
with open('parameters_learned.txt', 'w') as of:
    of.write(f"{params[0]}\n")
    of.write(f"{params[1]}\n")
    of.write(f"{params[2]}\n")
    of.write(f"{params[3]}\n")

# %%
# from hmmlearn import hmm

# model = hmm.GaussianHMM(2,
#                         'spherical',
#                         startprob_prior=stationary(P),
#                         transmat_prior=P,
#                         means_prior=means,
#                         covars_prior=np.square(stddevs),
#                         n_iter=10)
# model.fit(data.reshape(-1, 1))
# model.transmat_, model.means_, model.covars_

# %%
# with open('states_Viterbi_after_learning.txt', 'r') as of:
#     output = [1 if i == '"La Nina"\n' else 0 for i in of.readlines()]
# print(output == viterbi(data, params[0], params[1], np.sqrt(params[2])))
output(viterbi(data, params[0], params[1], np.sqrt(params[2])),
       'states_Viterbi_after_learning.txt')
