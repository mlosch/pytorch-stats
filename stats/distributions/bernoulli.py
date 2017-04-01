class Bernoulli(object):
    def __init__(self, p):
        if p < 0. or p > 1.:
            raise ValueError('parameter p must be in interval [0, 1]')
        self.p = p

    def __call__(self, *args):
        x = args[0]
        p = self.p.expand(x.size())

        return (p ** x) * ((1. - p) ** (1. - x))
