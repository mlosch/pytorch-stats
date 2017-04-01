import numpy as np
import torch
from torch.autograd import Variable
from stats.tensor import tensor


def fit(pdfs, parameters, observations, iter, lr):
    """Estimates the parameters of a mixture model via maximum likelihood maximization.
    Uses gradient descent for optimization.


    Parameters
    ----------
    pdfs :          List of callable pdfs
                    Callable probability density functions (likelihood function)
                    expecting an array of observations as the only argument.
    parameters :    List of list
                    List of list of parameters that are subject to optimization.
                    e.g. for a bimodal gaussian mixture: [[mu_1, sigma_1], [mu_2, sigma_2]]
    observations :  ndarray
                    Observations from an unknown pdf which parameters are subject to be estimated
    iter :          float
                    Maximum number of iterations
    lr :            float
                    Gradient descent learning rate

    Returns
    -------

    """

    # number of models/classes in mixture
    K = len(parameters)

    # initialize mixing coefficients with random values
    mixcoeffs = np.random.rand(K)
    mixcoeffs /= np.sum(mixcoeffs)

    # make the coefficients visible to the update step
    for k in range(K):
        mixcoeff = Variable(tensor(mixcoeffs[k]), requires_grad=True)
        parameters[k].append(mixcoeff)

    for i in range(iter):
        likelihood = 0
        for k in range(K):
            # multiply the likelihood with the mixing coefficients
            #  mixing coefficient: p(z_k = 1)
            p_z = parameters[k][-1].expand(observations.size())
            likelihood += pdfs[k](observations) * p_z

        expectation = torch.mean(torch.log(likelihood))

        # add constraint sum(mixcoeffs) = 1 via lagrange multiplier
        for k in range(K):
            expectation -= 1.0 * parameters[k][-1]
        expectation += 1.0  # c = 1

        if np.isnan(expectation.data[0]):
            raise RuntimeError('Singular state. Try different initial parameters')

        # Determine gradients
        expectation.backward()

        # Update parameters with gradient descent
        for k in range(K):
            for param in parameters[k]:
                param.data.add_(lr * param.grad.data)
                param.grad.data.zero_()

    return expectation.data[0]


if __name__ == '__main__':
    from stats.distributions import Normal
    """
    Estimate mean and std of a gaussian mixture model via MixtureModel-MLE on Kx10000 observations
    """

    np.random.seed(0)

    # number of gaussian models in mixture
    K = 2

    pdfs = []
    params = []
    true_params = []
    xs = []

    for k in range(K):
        # Sample observations from a bimodal normal distribution function with different parameter
        true_mean = np.random.uniform(-10, 10)
        true_std = np.random.uniform(0.5, 3.0)
        xs.append(true_mean + np.random.randn(np.random.randint(500, 2000)) * true_std)

        # Define likelihood function of model
        mean_estimate = Variable(tensor(true_mean+5.*np.random.randn()), requires_grad=True)
        std_estimate = Variable(tensor(1.0), requires_grad=True)

        pdfs.append(Normal(mean_estimate, std_estimate))

        params.append([mean_estimate, std_estimate])
        true_params.append([true_mean, true_std])

    x = np.concatenate(xs, axis=0)
    observations = Variable(tensor(x))

    log_likelihood = fit(pdfs, params, observations, iter=500, lr=0.1)

    print('Log likelihood: %7.5f' % log_likelihood)
    for k in range(K):
        print('k=%d mean=% 7.5f std=% 7.5f coeff=% 7.5f' % (k, params[k][0].data[0], params[k][1].data[0], params[k][2].data[0]))

    """
    Plot true and estimated distributions
    """

    import matplotlib.pyplot as plt
    n, _, _ = plt.hist(x, 100, normed=True)

    # plot distributions
    np_pdf = lambda x, mean, std: 1./np.sqrt(2.0*np.pi*std*std) * np.exp(- ((x-mean)**2) / (2.0 * std*std))
    xx = np.linspace(np.min(x), np.max(x), 1000)

    for k in range(K):
        true_y = np_pdf(xx, true_params[k][0], true_params[k][1])
        estimated_y = np_pdf(xx, params[k][0].data[0], params[k][1].data[0])

        plt.plot(xx, true_y, '-', label='Target pdf k=%d'%(k+1))
        plt.plot(xx, estimated_y, '-', label='Estimated pdf %d' % (k+1))

    plt.legend()

    plt.show()





