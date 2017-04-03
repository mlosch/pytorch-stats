import torch
from stats.distributions import Normal
from torch.autograd import Variable


class Polynomial(object):
    def __init__(self, w):
        assert w.dim() > 1
        self.n = w.numel()
        self.w = w
        pows = torch.linspace(0, self.n-1, self.n)

        if isinstance(w, Variable):
            pows = pows.type(w.data.type())
        else:
            pows = pows.type(w.type())

        self.pows = Variable(pows)

    def __call__(self, *args):
        X = args[0]
        assert (X.dim() > 1)

        sz = X.size()
        N = sz[0]

        Xrsz = [1]*(len(sz)-1) + [self.n]
        Xpow = torch.pow(X.repeat(*Xrsz), self.pows.repeat(N, 1))
        return torch.sum(torch.mm(Xpow, self.w), 1)


if __name__ == '__main__':
    from stats.tensor import tensor
    from stats.estimation import map, mle
    import numpy as np

    # pdf = Normal(Polynomial(torch.rand(3, 1)), torch.ones(1))
    # x = torch.linspace(-1, 1, 10).view(10, 1)
    # y = pdf(x)

    degree = 3 + 1

    w = Variable(tensor(np.random.randn(degree, 1)), requires_grad=True)
    std = Variable(tensor(1000.0), requires_grad=True)
    model = Polynomial(w)
    predictions = lambda args: model(args[1])
    pdf = Normal(predictions, std)

    w_prior = Normal(Variable(tensor(np.zeros((degree, 1)))), Variable(tensor([100]+[1]*(degree-1))))
    std_prior = Normal(Variable(tensor(0)), Variable(tensor(1)))
    prior = lambda args: torch.prod(w_prior(args[0])) * std_prior(args[1])

    # sample observations from true model and add noise
    true_model = Polynomial(Variable(tensor(np.random.randn(degree, 1))))
    sigma = 500.0
    N = 50
    x = tensor(np.linspace(-10, 10, N)).view(N, 1)
    y = true_model(Variable(x)).data
    noise = tensor(np.random.randn(N, 1)) * sigma
    observations = (y + noise).type(torch.cuda.DoubleTensor)

    # map.fit(pdf, prior, [w, std], (Variable(observations), Variable(x)))
    mle.fit(pdf, [w, std], (Variable(observations), Variable(x)))

    print(w.data)

    # plot
    import matplotlib.pyplot as plt

    npx = np.linspace(-10, 10, 100)
    thx = Variable(tensor(npx).view(100, 1))

    thy = true_model(thx)
    plt.plot(npx, true_model(thx).data.cpu().numpy(), '-.', label='True model')
    plt.plot(x.cpu().numpy(), observations.cpu().numpy(), '.', label='Observations')
    plt.plot(npx, model(thx).data.cpu().numpy(), '-', label='Estimate')
    plt.legend()

    plt.show()




