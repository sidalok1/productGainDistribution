import numpy as np
import matplotlib.pyplot as plt
import argparse

def notComment(string):
    return not (string.startswith("#"))

def reverseCumProd(array: np.ndarray) -> np.ndarray:
    return np.flip(np.cumprod(np.flip(array)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-file', default="data.txt", required=False)
    args = parser.parse_args()
    data = list()
    with open(args.file, 'r') as f:
        for line in filter(notComment, f.readlines()):
            line: str
            numbers = line.split(',')
            data.append(np.array(numbers, dtype=float))
    data = np.array(data)
    VALs = data[:,0]
    MUs = data[:,1]
    exp_R = reverseCumProd(MUs)
    SIGMAs = data[:,2]
    VARIANCEs = np.square(SIGMAs)
    var_R = reverseCumProd(np.square(SIGMAs) + np.square(MUs)) - reverseCumProd(np.square(MUs))

    mu = np.sum(exp_R * VALs)
    variance = np.sum(np.square(VALs) * var_R)
    sigma = variance**0.5

    def norm(x):
        return (1 / (2 * np.pi * variance)**0.5) * np.exp(-((x - mu) ** 2) / (2 * variance))

    domain = np.linspace((mu + (sigma * -4)), (mu + (sigma * 4)), 1000)

    dist = [norm(x) for x in domain]


    fig, ax = plt.subplots()
    ax.plot(domain, dist)
    ax.set_title(fr"Distribution with $\mu$ = {mu:.2f} and $\sigma$ = {sigma:.2f}")
    ax.set_xlabel("Account value")
    ax.set_ylabel("Probability")
    plt.savefig("distribution.png")