import torch


def centering(M):
    n = M.shape[0]
    mat_ones = torch.ones((n, n))
    idendity = torch.eye(n)
    H = idendity - mat_ones / n

    C = torch.matmul(M, H)
    return C


def gaussian_grammat(x, sigma2=None):
    xxT = torch.squeeze(torch.matmul(x, x.T))
    x2 = torch.diag(xxT)
    xnorm = x2 - xxT + (x2 - xxT).T

    if sigma2 is None:
        sigma2 = torch.median(xnorm[xnorm != 0])

    if sigma2 == 0:
        sigma2 += 1e-16

    Kx = torch.exp(-xnorm / sigma2)

    return Kx


def HSIC(x, y):
    gram_x = gaussian_grammat(x)
    gram_y = gaussian_grammat(y)

    c = x.shape[0] ** 2
    hsic = torch.trace(torch.matmul(centering(gram_x), centering(gram_y))) / c

    return hsic