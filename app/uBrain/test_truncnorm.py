import torch
from scipy.stats import truncnorm
import matplotlib.pyplot as plt


def truncated_normal(t, mean=0.0, std=0.01):
    torch.nn.init.normal_(t, mean=mean, std=std)
    while True:
        cond = torch.logical_or(t < mean - 2*std, t > mean + 2*std)
        if torch.sum(cond):
            t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
        else:
            break
    return t
    

def test_truncnorm():
    fig, ax = plt.subplots(1, 1)

    size = 1000000

    # a, b = -2, 2
    # r = truncnorm.rvs(a, b, size=size)
    # ax.hist(r, density=True, histtype='stepfilled', alpha=0.2, bins=50, label="scipy")

    tensor = torch.zeros(size)
    tensor = truncated_normal(tensor, std=0.05)
    r = tensor.numpy()
    ax.hist(r, density=True, histtype='stepfilled', alpha=0.2, bins=50, label="truncated_normal")

    tensor = torch.zeros(int(size/1000), int(size/1000))
    tensor = torch.nn.init.xavier_normal_(tensor)
    r = tensor.numpy().flatten()
    ax.hist(r, density=True, histtype='stepfilled', alpha=0.2, bins=50, label="xavier")

    tensor = torch.zeros(int(size/1000), int(size/1000))
    tensor = torch.nn.init.kaiming_normal_(tensor)
    r = tensor.numpy().flatten()
    ax.hist(r, density=True, histtype='stepfilled', alpha=0.2, bins=50, label="kaiming")

    ax.legend(loc='best', frameon=False)
    plt.show()


if __name__ == '__main__':
    test_truncnorm()