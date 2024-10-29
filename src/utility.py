
from tqdm import tqdm
from math import log
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import grad
from torch.quasirandom import SobolEngine

from botorch.optim import optimize_acqf

from botorch.models.transforms.outcome import Standardize
from botorch.utils.transforms import unnormalize
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.acquisition.utils import project_to_target_fidelity



import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import Interval

import warnings

from .acquisition import *
warnings.filterwarnings("ignore")


def optimize_acqf_and_get_acq(acq_func, batch_size=1, bounds=None, num_sample:int=512):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
    # optimize
    candidates, acq_value = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=batch_size,
        num_restarts=num_sample,
        raw_samples=num_sample,  # used for intialization heuristic
        options={"batch_limit": 100, "maxiter": 100},
    )
    return candidates, acq_value

def get_fitted_mlp(model, train_x, train_y, num_iter, optimizer):
    # train the model with the given data
    train_iter = tqdm(range(num_iter), 'Training MLP')
    model.train()
    for i in train_iter:
        model.zero_grad()
        pred = model(train_x)
        loss = F.mse_loss(pred, train_y)
        loss.backward()
        optimizer.step()
        train_iter.set_postfix(loss=loss.item())

def get_fitted_model(X, Y,training=True, normalize = False, max_cholesky_size=4096, **kwargs):
    '''
    Use Botorch API to fit a GP model
    '''
    dim = X.shape[-1]
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
    covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
        MaternKernel(nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0))
    )
    if normalize:
        model = SingleTaskGP(
            X,
            Y,
            covar_module=covar_module,
            likelihood=likelihood,
            outcome_transform=Standardize(m=1),
        )
    else:
        model = SingleTaskGP(
            X,
            Y,
            covar_module=covar_module,
            likelihood=likelihood,
        )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    if training:
        try:
            with gpytorch.settings.max_cholesky_size(max_cholesky_size):
                fit_gpytorch_mll(mll)
        except:
            pass

    return model

def generate_initial_data(problem, fidelities, n=16, **kwargs):
    # generate training data
    dim = problem.dim
    if hasattr(problem, 'candidates'):
        if n <= len(problem.candidates):
            choise = np.random.choice(len(problem.candidates), n, replace=False)
        else:
            choise = np.random.choice(len(problem.candidates), n, replace=True)
        train_x_full = problem.candidates[choise]
        train_obj = problem.objectives[choise].reshape([n, 1])
    else:
        train_x = unnormalize(torch.rand(n, dim-1, **kwargs), bounds=problem.bounds[:,:-1])
        train_f = fidelities[torch.randint(len(fidelities), (n, 1))]
        train_x_full = torch.cat((train_x, train_f), dim=1)
        train_obj = problem(train_x_full).unsqueeze(-1)  # add output dimension
    return train_x_full, train_obj

def get_project(target_fidelities):
    def project(X):
        return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)
    return project

def initialize_mf_model(train_x, train_obj, data_fidelity:int):
    # define a surrogate model suited for a "training data"-like fidelity parameter
    # in dimension 6, as in [2]
    model = SingleTaskMultiFidelityGP(
        train_x, train_obj, 
        outcome_transform=Standardize(m=1), 
        data_fidelity=data_fidelity
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model

def compute_maximum_posterior_variance(model, beta: Tensor, bounds=None)->float:
    '''
    Compute the maximum mutual information for a given model
    Args:
    - model: the GP model to compute the information
    - beta: the beta value
    - bounds: the bounds of the optimization space
    '''
    _acq_ci = qConfidenceInterval(model, beta)
    _, ci_width = optimize_acqf_and_get_acq(_acq_ci, batch_size=1, bounds=bounds, num_sample=512)
    return ci_width.item()

def compute_information_bp_fast_classification(model, x_tr, y_tr, batch_size=200, no_bp = False):
        """Compute the full information with back propagation support.
        Using delta_w.T gw @ gw.T delta_w = (delta_w.T gw)^2 for efficient computation.
        Args:
            no_bp: detach the information term hence it won't be used for learning
        Cite: https://github.com/RyanWangZf/PAC-Bayes-IB/blob/main/src/models.py
        """
        def one_hot_transform(y, num_class=100):
            one_hot_y = F.one_hot(y, num_classes=model.num_classes)
            return one_hot_y.float()

        all_tr_idx = np.arange(len(x_tr))
        np.random.shuffle(all_tr_idx)

        num_all_batch = int(np.ceil(len(x_tr)/batch_size))

        param_keys = [p[0] for p in model.named_parameters()]
        delta_w_dict = dict().fromkeys(param_keys)
        for pa in model.named_parameters():
            if "weight" in pa[0]:
                w0 = model.w0_dict[pa[0]]
                delta_w = pa[1] - w0
                delta_w_dict[pa[0]] = delta_w

        info_dict = dict()
        gw_dict = dict().fromkeys(param_keys)

        for idx in range(10):
            # print("compute emp fisher:", idx)
            sub_idx = np.random.choice(all_tr_idx, batch_size)
            x_batch = x_tr[sub_idx]
            y_batch = y_tr[sub_idx]

        for idx in range(10):
            # print("compute emp fisher:", idx)
            sub_idx = np.random.choice(all_tr_idx, batch_size)
            x_batch = x_tr[sub_idx]
            y_batch = y_tr[sub_idx]

            y_oh_batch = one_hot_transform(y_batch, model.num_class)
            pred = model.forward(x_batch)
            loss = F.cross_entropy(pred, y_batch,
                        reduction="mean")

            gradients = grad(loss, model.parameters())
            
            for i, gw in enumerate(gradients):
                gw_ = gw.flatten()
                if gw_dict[param_keys[i]] is None:
                    gw_dict[param_keys[i]] = gw_
                else:
                    gw_dict[param_keys[i]] += gw_
        
        for k in gw_dict.keys():
            if "weight" in k:
                gw_dict[k] *= 1/num_all_batch
                delta_w = delta_w_dict[k]
                # delta_w.T gw @ gw.T delta_w = (delta_w.T gw)^2
                info_ = (delta_w.flatten() * gw_dict[k]).sum() ** 2
                if no_bp:
                    info_dict[k] = info_.item()
                else:
                    info_dict[k] = info_

        return info_dict

def compute_information_bp_fast_regression(model, loss, x_tr, y_tr, batch_size=200, no_bp = False):
        """Compute the full information with back propagation support.
        Using delta_w.T gw @ gw.T delta_w = (delta_w.T gw)^2 for efficient computation.
        Args:
            model: the model to compute the information
            loss: the loss function to compute the gradient
            x_tr: the training data
            y_tr: the training target
            batch_size: the batch size for computing the gradient
            no_bp: detach the information term hence it won't be used for learning
        Cite: https://github.com/RyanWangZf/PAC-Bayes-IB/blob/main/src/models.py
        """
        all_tr_idx = np.arange(len(x_tr))
        np.random.shuffle(all_tr_idx)

        num_all_batch = int(np.ceil(len(x_tr)/batch_size))

        param_keys = [p[0] for p in model.named_parameters()]
        delta_w_dict = dict().fromkeys(param_keys)
        for pa in model.named_parameters():
            if "weight" in pa[0]:
                w0 = model.w0_dict[pa[0]]
                delta_w = pa[1] - w0
                delta_w_dict[pa[0]] = delta_w

        info_dict = dict()
        gw_dict = dict().fromkeys(param_keys)

        for idx in range(10):
            sub_idx = np.random.choice(all_tr_idx, batch_size)
            x_batch = x_tr[sub_idx]
            y_batch = y_tr[sub_idx]

            pred = model.forward(x_batch)
            # loss = F.cross_entropy(pred, y_batch,
            #             reduction="mean")

            gradients = grad(loss, model.parameters())
            
            for i, gw in enumerate(gradients):
                gw_ = gw.flatten()
                if gw_dict[param_keys[i]] is None:
                    gw_dict[param_keys[i]] = gw_
                else:
                    gw_dict[param_keys[i]] += gw_
        
        for k in gw_dict.keys():
            if "weight" in k:
                gw_dict[k] *= 1/num_all_batch
                delta_w = delta_w_dict[k]
                # delta_w.T gw @ gw.T delta_w = (delta_w.T gw)^2
                info_ = (delta_w.flatten() * gw_dict[k]).sum() ** 2
                if no_bp:
                    info_dict[k] = info_.item()
                else:
                    info_dict[k] = info_

        return info_dict

def monte_carlo_entropy(n_samples:torch.tensor):
    """
    Estimate the entropy of a single variable using Monte Carlo sampling.

    Parameters:
    - n_samples: The number of samples to draw.

    Returns:
    - Estimated entropy.
    """
    entropy_estimate = torch.zeros(1)
    total_num = n_samples.size(0)
    _, counts = torch.unique(n_samples, return_counts=True)
    for c in counts:
        entropy_estimate -= c/total_num * torch.log(c/total_num)

    return entropy_estimate

def monte_carlo_excessive_risk(fidelity_counts:torch.tensor, beta:torch.tensor)->torch.tensor:
    """
    Estimate the excessive risk of a single variable using Monte Carlo sampling.

    Parameters:
    - fidelity_counts: The number of samples at each fidelity.
    - beta: The beta value to balance risk and regret bound

    Returns:
    - Estimated excessive risk.
    """
    # entropy = monte_carlo_entropy(n_samples)
    excessive_risk_estimate = torch.ones_like(fidelity_counts, dtype=torch.float)
    for i, c in enumerate(fidelity_counts):
        if c == 0:
            c = torch.ones(1, dtype=torch.float) * 0.5 # avoid division by zero
        if i == 0:
            excessive_risk_estimate[i] = 1/ torch.sqrt(c) * beta
            continue

        previous_risk = excessive_risk_estimate[i-1] if excessive_risk_estimate[i-1] < float('inf') else torch.ones(1, dtype=torch.float)
        if previous_risk < 1:
            excessive_risk_estimate[i] = previous_risk
        else:
            excessive_risk_estimate[i] = beta * (previous_risk + torch.sqrt(torch.log(previous_risk))/c)

    # return excessive_risk_estimate[-1] * torch.sqrt(entropy) * 2
    return excessive_risk_estimate[-1]

def excessive_risk_reduction_rate(fidelity_counts:torch.tensor, fidelity_choosen:int, beta:torch.tensor)->torch.tensor:
    """
    Compute the risk reduction rate of the excessive risk.
    Assuming the entropy doesn't change much

    Parameters:
    - fidelity_counts: The number of samples at each fidelity.
    - beta: The beta value to balance risk and regret bound
    - fidelity_choosen: The fidelity choosen to compute the risk reduction rate

    Returns:
    - Risk reduction rate.
    """
    new_fidelity_counts = fidelity_counts.clone()
    new_fidelity_counts[fidelity_choosen] += 1
    rate = monte_carlo_excessive_risk(fidelity_counts, beta) - monte_carlo_excessive_risk(new_fidelity_counts, beta)
    return rate

def rbf_kernel_variance_reduction_rate(T:int, dim:int, variance:torch.tensor)->torch.tensor:
    """
    Compute the variance reduction rate of the RBF kernel.

    Parameters:
    - T: The number of iterations.
    - dim: The dimension of the input space.
    - Variance: current maximum variance

    Returns:
    - Variance reduction rate.
    """
    rate = 1 - log(T) ** (dim + 1) / log(T+1) ** (dim + 1)
    return variance * rate

