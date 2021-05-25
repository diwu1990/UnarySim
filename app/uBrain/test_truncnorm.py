import torch
from scipy.stats import truncnorm
import matplotlib.pyplot as plt


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2*std) & (tmp > -2*std)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def truncated_normal(t, mean=0.0, std=0.01):
    torch.nn.init.normal_(t, mean=mean, std=std)
    while True:
        cond = torch.logical_or(t < mean - 2*std, t > mean + 2*std)
        if torch.sum(cond):
            t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
        else:
            break
    return t
    

fig, ax = plt.subplots(1, 1)


def test_truncnorm():
    a, b = -2, 2
    size = 10000000
    r = truncnorm.rvs(a, b, size=size)
    ax.hist(r, density=True, histtype='stepfilled', alpha=0.2, bins=50, label="scipy")

    tensor = torch.zeros(size)
    truncated_normal_(tensor, std=1)
    r = tensor.numpy()
    ax.hist(r, density=True, histtype='stepfilled', alpha=0.2, bins=50, label="custom")

    tensor = torch.zeros(size)
    tensor = truncated_normal(tensor, std=1)
    r = tensor.numpy()
    ax.hist(r, density=True, histtype='stepfilled', alpha=0.2, bins=50, label="custom2")

    ax.legend(loc='best', frameon=False)
    plt.show()


if __name__ == '__main__':
    test_truncnorm()