import numpy as np
import torch
from torch.autograd import Variable
from stats import map
from stats.tensor import tensor


def fit(func, parameters, observations, iter=1000, lr=0.1):
    """Estimates the parameters of an arbitrary function via maximum likelihood estimation and
    uses plain old gradient descent for optimization


    Parameters
    ----------
    func :          Callable pdf
                    Callable probability density function (likelihood function)
                    expecting an array of observations as the only argument.
    parameters :    List
                    List of parameters that are subject to optimization.
    observations :  ndarray
                    Observations from an unknown pdf which parameters are subject to be estimated
    iter :          float
                    Maximum number of iterations
    lr :            float
                    Gradient descent learning rate

    Returns
    -------

    """

    # Use MAP with uniform prior
    prior_ = Variable(tensor(1.0))
    return map.fit(func, lambda x: prior_, parameters, observations, iter, lr)

    # Explicit implementation without prior:
    # for i in range(iter):
    #     # Define objective function (log-likelihood) to maximize
    #     likelihood = torch.mean(torch.log(func(observations)))
    #
    #     # Determine gradients
    #     likelihood.backward()
    #
    #     # Update parameters with gradient descent
    #     for param in parameters:
    #         param.data.add_(lr * param.grad.data)
    #         param.grad.data.zero_()


def normal_pdf(x, mean, std):
    mean = mean.expand(x.size())
    var = std.expand(x.size()) ** 2
    return 1./torch.sqrt(2.0*np.pi*var) * torch.exp(- ((x-mean)**2) / (2.0 * var))


if __name__ == '__main__':
    """
    Estimate mean and std of a normal distribution via MLE on 10000 observations
    """

    mean = Variable(tensor(0), requires_grad=True)
    std = Variable(tensor(1), requires_grad=True)

    func = lambda x: normal_pdf(x, mean, std)

    # Sample observations from a normal distribution function with different parameter
    true_mean = np.random.rand()
    true_std = np.random.rand()
    x = true_mean + np.random.randn(10000) * true_std
    x = Variable(tensor(x))

    fit(func, [mean, std], x)

    print('Estimated parameter: {{{}, {}}}, True parameter: {{{}, {}}}'.format(mean.data[0], std.data[0], true_mean, true_std))

