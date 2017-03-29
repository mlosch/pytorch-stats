import numpy as np
import torch
from torch.autograd import Variable


def fit(func, prior, parameters, observations, iter=1000, lr=0.1):
    """Estimates the parameters of an arbitrary function via maximum a posteriori estimation and
    uses plain old gradient descent for optimization


    Parameters
    ----------
    func :          Callable pdf
                    Callable probability density function (likelihood function)
                    expecting an array of observations as the only argument.
                    e.g. p(x|params) = func(observations)
    prior :         Callable pdf
                    Callable probability density function over parameters
                    expecting an array of parameters as the only argument.
                    e.g. p(params) = prior(parameters)
                    if p(params) is a uniform distribution, this method equals MLE
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

    for i in range(iter):
        # Define objective function (log-likelihood) to maximize
        prior_ = torch.log(prior(parameters))
        posterior = torch.mean(torch.log(func(observations))) + prior_

        if np.isnan(posterior.data[0]) or np.isnan(prior_.data[0]):
            return

        # Determine gradients
        posterior.backward()

        # Update parameters with gradient descent
        for param in parameters:
            param.data.add_(lr * param.grad.data)
            param.grad.data.zero_()


def normal_pdf(x, mean, std):
    mean = mean.expand(x.size())
    var = std.expand(x.size()) ** 2
    p = 1./torch.sqrt(2.0*np.pi*var) * torch.exp(- ((x-mean)**2) / (2.0 * var))
    return p


def parameter_prior(args, mean_pdf, std_pdf):
    mean, std = args
    mean_prior = mean_pdf(mean)
    std_prior = std_pdf(std)
    return mean_prior * std_prior


def tensor(arr, dtype=torch.cuda.DoubleTensor):
    return torch.from_numpy(np.array(arr)).type(dtype)


if __name__ == '__main__':
    """
    Estimate mean and std of a normal distribution via MAP on 2x10000 observations sampled from a bimodal distribution
    """
    dtype = torch.cuda.DoubleTensor

    mean_estimate = Variable(tensor([0.0]), requires_grad=True)
    std_estimate = Variable(tensor([1.0]), requires_grad=True)

    # Sample observations from a bimodal normal distribution function with different parameter
    true_mean = (np.random.rand()-0.5) * 5.0
    true_std = np.random.rand() * 10.0
    x = true_mean + np.random.randn(10000) * true_std

    outlier_mean = true_mean + 3.0
    outlier_std = max(0.3, np.random.rand())
    x = np.concatenate([x, outlier_mean + np.random.randn(10000) * outlier_std])

    # Define likelihood function of model
    func = lambda x: normal_pdf(x, mean_estimate, std_estimate)

    # Define prior over mean and std
    #  Let the std_prior have the true value
    mean_prior = lambda x: normal_pdf(x, Variable(tensor([0.0])), Variable(tensor([10.0])))
    std_prior = lambda x: normal_pdf(x, Variable(tensor([true_std])), Variable(tensor([1.0])))
    prior = lambda args: parameter_prior(args, mean_prior, std_prior)

    xvar = Variable(tensor(x))

    fit(func, prior, [mean_estimate, std_estimate], xvar, iter=5000, lr=0.001)

    print('Estimated parameter: {{{}, {}}}, True parameter: {{{}, {}}}'.format(mean_estimate.data[0], std_estimate.data[0], true_mean, true_std))
    print('Distractor distribution parameter: {{{}, {}}}'.format(outlier_mean, outlier_std))

    """
    Plot true and estimated distributions
    """

    import matplotlib.pyplot as plt
    n, _, _ = plt.hist(x, 100, normed=True)

    # plot distributions
    np_pdf = lambda x, mean, std: 1./np.sqrt(2.0*np.pi*std*std) * np.exp(- ((x-mean)**2) / (2.0 * std*std))
    xx = np.linspace(np.min(x), np.max(x), 1000)
    true_y = np_pdf(xx, true_mean, true_std)
    outlier_y = np_pdf(xx, outlier_mean, outlier_std)
    estimate_y = np_pdf(xx, mean_estimate.data[0], std_estimate.data[0])
    mle_y = np_pdf(xx, x.mean(), x.std())

    plt.plot(xx, true_y, 'g-', label='Target model')
    plt.plot(xx, outlier_y, 'r-', label='Distractor model')
    plt.plot(xx, estimate_y, 'b-', label='Estimated model')
    plt.plot(xx, mle_y, 'k-', label='MLE')
    plt.legend()

    plt.show()

