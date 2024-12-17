#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    ANFIS in torch: some fuzzy membership functions.
    @author: James Power <james.power@mu.ie> Apr 12 18:13:10 2019
"""

import enum
import torch
import numpy as np
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
#from anfis import AnfisNet
from anfis_Mandami import AnfisNet


class MfsType(enum.Enum):
    Sigmoid = 0
    DSigmoid = 1
    Gauss = 2
    Bell = 3
    Triangular = 4
    Trapezoid = 5
    
    def __str__(self):
        return self.name


def _mk_param(val):
    '''Make a torch parameter from a scalar value'''
    if isinstance(val, torch.Tensor):
        val = val.item()
    return torch.nn.Parameter(torch.tensor(val, dtype=torch.float))


class GaussMembFunc(torch.nn.Module):
    '''
        Gaussian membership functions, defined by two parameters:
            mu, the mean (center)
            sigma, the standard deviation.
    '''
    def __init__(self, mu, sigma):
        super(GaussMembFunc, self).__init__()
        self.register_parameter('mu', _mk_param(mu))
        self.register_parameter('sigma', _mk_param(sigma))

    def forward(self, x):
        val = torch.exp(-torch.pow(x - self.mu, 2) / (2 * self.sigma**2))
        return val

    def pretty(self):
        return 'GaussMembFunc {} {}'.format(self.mu, self.sigma)


def make_gauss_mfs(sigma, mu_list):
    '''Return a list of gaussian mfs, same sigma, list of means'''
    return [GaussMembFunc(mu, sigma) for mu in mu_list]


class BellMembFunc(torch.nn.Module):
    '''
        Generalised Bell membership function; defined by three parameters:
            a, the half-width (at the crossover point)
            b, controls the slope at the crossover point (which is -b/2a)
            c, the center point
    '''
    def __init__(self, a, b, c):
        super(BellMembFunc, self).__init__()
        self.register_parameter('a', _mk_param(a))
        self.register_parameter('b', _mk_param(b))
        self.register_parameter('c', _mk_param(c))
        self.b.register_hook(BellMembFunc.b_log_hook)

    @staticmethod
    def b_log_hook(grad):
        '''
            Possibility of a log(0) in the grad for b, giving a nan.
            Fix this by replacing any nan in the grad with ~0.
        '''
        grad[torch.isnan(grad)] = 1e-9
        return grad

    def forward(self, x):
        dist = torch.pow((x - self.c)/self.a, 2)
        return torch.reciprocal(1 + torch.pow(dist, self.b))

    def pretty(self):
        return 'BellMembFunc {} {} {}'.format(self.a, self.b, self.c)


def make_bell_mfs(a, b, clist):
    '''Return a list of bell mfs, same (a,b), list of centers'''
    return [BellMembFunc(a, b, c) for c in clist]


class TriangularMembFunc(torch.nn.Module):
    '''
        Triangular membership function; defined by three parameters:
            a, left foot, mu(x) = 0
            b, midpoint, mu(x) = 1
            c, right foot, mu(x) = 0
    '''
    def __init__(self, a, b, c):
        super(TriangularMembFunc, self).__init__()
        assert a <= b and b <= c,\
            'Triangular parameters: must have a <= b <= c.'
        self.register_parameter('a', _mk_param(a))
        self.register_parameter('b', _mk_param(b))
        self.register_parameter('c', _mk_param(c))

    @staticmethod
    def isosceles(width, center):
        '''
            Construct a triangle MF with given width-of-base and center
        '''
        return TriangularMembFunc(center-width, center, center+width)

    def forward(self, x):
        '''
        return torch.where(
            torch.ByteTensor(self.a < x) & torch.ByteTensor(x <= self.b),
            (x - self.a) / (self.b - self.a),
            # else
            torch.where(
                torch.ByteTensor(self.b < x) & torch.ByteTensor(x <= self.c),
                (self.c - x) / (self.c - self.b),
                torch.zeros_like(x, requires_grad=True)))
        '''
        return torch.where(
            (self.a < x).byte() & (x <= self.b).byte(),
            (x - self.a) / (self.b - self.a),
            # else
            torch.where(
                (self.b < x).byte() & (x <= self.c).byte(),
                (self.c - x) / (self.c - self.b),
                torch.zeros_like(x, requires_grad=True)))

    def pretty(self):
        return 'TriangularMembFunc {} {} {}'.format(self.a, self.b, self.c)


def make_tri_mfs(width, clist):
    '''Return a list of triangular mfs, same width, list of centers'''
    return [TriangularMembFunc(c-width/2, c, c+width/2) for c in clist]


class TrapezoidalMembFunc(torch.nn.Module):
    '''
        Trapezoidal membership function; defined by four parameters.
        Membership is defined as:
            to the left of a: always 0
            from a to b: slopes from 0 up to 1
            from b to c: always 1
            from c to d: slopes from 1 down to 0
            to the right of d: always 0
    '''
    def __init__(self, a, b, c, d):
        super(TrapezoidalMembFunc, self).__init__()
        assert a <= b and b <= c and c <= d,\
            'Trapezoidal parameters: must have a <= b <= c <= d.'
        self.register_parameter('a', _mk_param(a))
        self.register_parameter('b', _mk_param(b))
        self.register_parameter('c', _mk_param(c))
        self.register_parameter('d', _mk_param(d))

    @staticmethod
    def symmetric(topwidth, slope, midpt):
        '''
            Make a (symmetric) trapezoid mf, given
                topwidth: length of top (when mu == 1)
                slope: extra length at either side for bottom
                midpt: center point of trapezoid
        '''
        b = midpt - topwidth / 2
        c = midpt + topwidth / 2
        return TrapezoidalMembFunc(b - slope, b, c, c + slope)

    @staticmethod
    def rectangle(left, right):
        '''
            Make a Trapezoidal MF with vertical sides (so a==b and c==d)
        '''
        return TrapezoidalMembFunc(left, left, right, right)

    @staticmethod
    def triangle(left, midpt, right):
        '''
            Make a triangle-shaped MF as a special case of a Trapezoidal MF.
            Note: this may revert to general trapezoid under learning.
        '''
        return TrapezoidalMembFunc(left, midpt, midpt, right)

    def forward(self, x):
        yvals = torch.zeros_like(x)
        if self.a < self.b:
            incr = torch.ByteTensor(self.a < x) & torch.ByteTensor(x <= self.b)
            yvals[incr] = (x[incr] - self.a) / (self.b - self.a)
        if self.b < self.c:
            decr = torch.ByteTensor(self.b < x) & torch.ByteTensor(x < self.c)
            yvals[decr] = 1
        if self.c < self.d:
            decr = torch.ByteTensor(self.c <= x) & torch.ByteTensor(x < self.d)
            yvals[decr] = (self.d - x[decr]) / (self.d - self.c)
        return yvals

    def pretty(self):
        return 'TrapezoidalMembFunc a={} b={} c={} d={}'.\
            format(self.a, self.b, self.c, self.d)


def make_trap_mfs(width, slope, clist):
    '''Return a list of symmetric Trap mfs, same (w,s), list of centers'''
    return [TrapezoidalMembFunc.symmetric(width, slope, c) for c in clist]


class SigmoidMembFunc(torch.nn.Module):
    '''
        Sigmoid membership function
    '''
    def __init__(self, alpha, c):
        super(SigmoidMembFunc, self).__init__()
        self.register_parameter('alpha', _mk_param(alpha))
        self.register_parameter('c', _mk_param(c))

    def forward(self, x):
        '''
        Calculate membership value using the sigmoid formula:
        u(x) = 1 / (1 + exp(-alpha * (x - c)))
        '''
        return torch.sigmoid(self.alpha * (x - self.c))
    
    def pretty(self):
        return 'SigmoidMembFunc alpha={} c={}'.format(self.alpha.item(), self.c.item())
    

def make_sigmoid_mfs(alpha, c_list):
    '''
        Return a list of sigmoid membership functions
    '''
    return [SigmoidMembFunc(alpha, c) for c in c_list]


class DsigMembFunc(torch.nn.Module):
    '''
        Difference of two sigmoids; defined by two centers and slopes.
    '''
    def __init__(self, alpha, beta, c1, c2):
        super(DsigMembFunc, self).__init__()
        self.register_parameter('alpha', _mk_param(alpha))
        self.register_parameter('beta', _mk_param(beta))
        self.register_parameter('c1', _mk_param(c1))
        self.register_parameter('c2', _mk_param(c2))

    def forward(self, x):
        '''
            u(x) = sigmoid(alpha * (x - c1)) - sigmoid(beta * (x - c2))
        '''
        return torch.sigmoid(self.alpha * (x - self.c1)) - torch.sigmoid(self.beta * (x - self.c2))
    
    def pretty(self):
        return f"DsigMembFunc alpha={self.alpha.item()}, beta={self.beta.item()}, c1={self.c1.item()}, c2={self.c2.item()}"
    

def make_dsig_mfs(alpha, beta, clist1, clist2):
    return [DsigMembFunc(alpha, beta, c1, c2) for c1, c2 in zip(clist1, clist2)]


# Make the classes available via (controlled) reflection:
get_class_for = {n: globals()[n]
                 for n in ['BellMembFunc',
                           'GaussMembFunc',
                           'TriangularMembFunc',
                           'TrapezoidalMembFunc',
                           ]}


def make_mfs(num_mfs, minval, maxval, range, mfs_type):
    if mfs_type == MfsType.Sigmoid:
        alpha = 10 / range
        clist = torch.linspace(minval, maxval, num_mfs).tolist()
        return make_sigmoid_mfs(alpha, clist)
    elif mfs_type == MfsType.Bell:
        a = range / num_mfs
        b = 2  # Slope control (arbitrary default)
        clist = torch.linspace(minval, maxval, num_mfs).tolist()
        return make_bell_mfs(a, b, clist)
    elif mfs_type == MfsType.Triangular:
        width = range / num_mfs
        clist = torch.linspace(minval, maxval, num_mfs).tolist()
        return make_tri_mfs(width, clist)
    elif mfs_type == MfsType.Trapezoid:
        slope = range / (2 * num_mfs)  # Example slope
        width = range / num_mfs
        clist = torch.linspace(minval, maxval, num_mfs).tolist()
        return make_trap_mfs(width, slope, clist)
    elif mfs_type == MfsType.DSigmoid:
        alpha = 10 / range  # Control the steepness of the sigmoids
        beta = alpha
        clist1 = torch.linspace(minval, maxval - range / num_mfs, num_mfs).tolist()
        clist2 = torch.linspace(minval + range / num_mfs, maxval, num_mfs).tolist()
        return make_dsig_mfs(alpha, beta, clist1, clist2)
    elif mfs_type == MfsType.Gauss or not mfs_type:
        sigma = range / num_mfs
        mulist = torch.linspace(0, maxval, num_mfs).tolist()
        return make_gauss_mfs(sigma, mulist)
    else:
        assert False,\
            f"Unsupported MF {mfs_type}"


def make_anfis(x, device, num_mfs=5, num_out=1, hybrid=True, mfs_type=MfsType.Gauss):
    '''
        Make an ANFIS model, auto-calculating the (Gaussian) MFs.
        I need the x-vals to calculate a range and spread for the MFs.
        Variables get named x0, x1, x2,... and y0, y1, y2 etc.
    '''

    #print(x.T[0])
    #print(x.T[1])
    #print(x.shape[1])

    n_terms = num_mfs
    num_invars = x.shape[1]
    minvals, _ = torch.min(x, dim=0)
    maxvals, _ = torch.max(x, dim=0)
    ranges = maxvals-minvals
    invars = []
    for i in range(num_invars):
        invars.append(('x{}'.format(i), make_mfs(num_mfs, minvals[i], maxvals[i], ranges[i], mfs_type)))
        #print(invars)
        #exit()
    outvars = ['y{}'.format(i) for i in range(num_out)]
    #model = AnfisNet('Simple classifier', invars, outvars, hybrid=hybrid)#versione originale
    model = AnfisNet('Simple classifier', invars, outvars, n_terms, device, hybrid=hybrid)
    #import experimental
    #experimental.plot_all_mfs(model, x)
    return model


def make_anfis_T(x, num_mfs=5, num_out=1, hybrid=True):
    '''
        Make an ANFIS model, auto-calculating the (Gaussian) MFs.
        I need the x-vals to calculate a range and spread for the MFs.
        Variables get named x0, x1, x2,... and y0, y1, y2 etc.
    '''
    n_terms = num_mfs

    num_invars = x.shape[1]
    minvals, _ = torch.min(x, dim=0)
    maxvals, _ = torch.max(x, dim=0)
    ranges = maxvals-minvals
    invars = []

    i = 0
    while i < num_invars:

        diff = minvals.numpy() + (maxvals.numpy() - minvals.numpy()) / 2
        prova_min = minvals.numpy() + (diff - minvals.numpy()) / 2
        prova_max = diff + (maxvals.numpy() - diff) / 2
        clist = (prova_min[i], diff[i], prova_max[i])
        ranges = diff[i] - minvals[i].numpy()
        invars.append(('x{}'.format(i), make_tri_mfs(ranges, clist)))

        i = i + 1

    outvars = ['y{}'.format(i) for i in range(num_out)]

    model = AnfisNet('Simple classifier', invars, outvars, n_terms, hybrid=hybrid)
    return model