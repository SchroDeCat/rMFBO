'''
# Define a deep kernel GP model
# define feature extractor
'''

import gpytorch
import torch
import tqdm
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize

class twoBranchMFModel(torch.nn.Module):

    def __init__(self, data_dim:int=1, hidden_dim:int=128):
        '''
        Assume for multi-fidelity input, the last dimension is the fidelity level
        '''
        super(twoBranchMFModel, self).__init__()
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        # spatial only branch
        self.branch1 = torch.nn.Sequential(
            torch.nn.Linear(data_dim-1, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU()
        )

        # fidelity correctioin branch
        self.branch2 = torch.nn.Sequential(
            torch.nn.Linear(data_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.LeakyReLU()
        )

        # top layer
        # self.linear3 = torch.nn.Linear((hidden_dim)*2, 1)
        self.linear3 = torch.nn.Linear((hidden_dim), 1)

    def forward(self, x):
        '''
        Assume x is a tensor with shape n * [d + 1]
        '''
        # print(x.size())
        # z1 = self.branch1(x[:,:self.data_dim-1])
        z2 = self.branch2(x)
        # z = z1 + z2
        # z = torch.cat([z1.reshape(-1, self.hidden_dim), z2.reshape(-1, self.hidden_dim)], dim=-1)
        # y = self.linear3(z)
        y = z2
        return y
    
class MFGPRegressionModel(SingleTaskGP):
        ''''
        A deep kernel GP model for multi-fidelity data
        '''
        def __init__(self, train_x, train_y, likelihood, trained_branch_mlp:torch.nn.Module=None):
            super(MFGPRegressionModel, self).__init__(train_x, train_y, likelihood)
            twoBranchMFModel = twoBranchMFModel(train_x.size(-1)) if trained_branch_mlp is None else trained_branch_mlp
            class MFFeastureExtractor(torch.nn.Module):
                '''
                A deep feature extractor extracting features from multi-fidelity data
                '''
                def __init__(self, model):
                    super(MFFeastureExtractor, self).__init__()
                    self.branch1 = model.branch1
                    self.branch2 = model.branch2

                def forward(self, x):
                    z1 = self.branch1(x[:, :-1])
                    z2 = self.branch2(x)
                    z = z1 + z2
                    return z

            self.feature_extractor = MFFeastureExtractor(model=twoBranchMFModel)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=2)
            )

            # This module will scale the NN features so that they're nice values
            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)
            self.to(train_x)

        def forward(self, x):
            # # We're first putting our data through a deep net (feature extractor)
            projected_x = self.feature_extractor(x)
            projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"
            return super().forward(projected_x)
      
class LargeFeatureExtractor(torch.nn.Sequential):
    '''
    A deep feature extractor
    '''
    def __init__(self, data_dim=1):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(data_dim, 128, dtype=torch.float64))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear4', torch.nn.Linear(128, max(data_dim-1, 2), dtype=torch.float64))

class RBFGPRegressionModel(SingleTaskGP):
    ''''
    A deep kernel GP model
    '''
    def __init__(self, train_x, train_y, likelihood, mean, covar):
        super(RBFGPRegressionModel, self).__init__(train_x, train_y, likelihood, outcome_transform=Standardize(m=1))
        self.mean_module = mean
        self.covar_module = covar

class GPRegressionModel(SingleTaskGP):
        ''''
        A deep kernel GP model
        '''
        def __init__(self, train_x, train_y, likelihood):
            super(GPRegressionModel, self).__init__(train_x, train_y, likelihood, outcome_transform=Standardize(m=1))
            feature_extractor = LargeFeatureExtractor(train_x.size(-1))
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=max(train_x.size(-1)-1,2))
            )
            self.feature_extractor = feature_extractor

            # This module will scale the NN features so that they're nice values
            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)
            self.to(train_x)

        def forward(self, x):
            # # We're first putting our data through a deep net (feature extractor)
            projected_x = self.feature_extractor(x)
            projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"
            return super().forward(projected_x)
        
def spectral_norm(weight, n_power_iterations=1):
    w = weight # avoid lossing gradient
    height = w.size(0)
    u = torch.randn(height).to(w)
    
    for _ in range(n_power_iterations):
        v = torch.mv(torch.t(w), u)
        v = v / torch.norm(v)
        u = torch.mv(w, v)
        u = u / torch.norm(u)
    
    sigma = torch.dot(u, torch.mv(w, v))
    return sigma

class SpectralRegularizer:
    '''
    Approximate the spectral norm of the weight matrix
    '''
    def __init__(self, model, coeff=1e-4, n_power_iterations=10):
        self.model = model
        self.coeff = coeff
        self.n_power_iterations = n_power_iterations

    def __call__(self):
        reg_loss = 0
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                sigma = spectral_norm(param, self.n_power_iterations)
                reg_loss += sigma
                # print(f"{name}: {sigma} {param.requires_grad}")
        return self.coeff * ((reg_loss - 1)**2)

def init_model_srdk(train_x, train_obj, coeff=1e-4, training_iter=1000, lr=1e-2, power_iterations=100, verbose=True, **kwargs):
    '''
    Initialize a deep kernel GP model with spectral regularization
    '''
    likelihood_c = gpytorch.likelihoods.GaussianLikelihood()
    model = GPRegressionModel(train_x, train_obj, likelihood_c)
    model.to(train_x)

    # Train the model
    model.train()
    likelihood_c.train()

    # Use the adam optimizer
    optimizer_c = torch.optim.Adam([
        {'params': model.feature_extractor.parameters()},
        {'params': model.covar_module.parameters()},
        # {"params": model.covar_module.base_kernel.parameters()},
        {'params': model.mean_module.parameters()},
        {'params': model.likelihood.parameters()},
    ], lr=lr)
    train_loss = kwargs.get('train_loss', "mae")
    if train_loss == "mll":
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    elif train_loss == "mae":
        mll = torch.nn.L1Loss()
    elif train_loss == "mse":
        mll = torch.nn.MSELoss()
    else:
        raise ValueError("Unknown train_loss")

    num_epochs = training_iter
    train_iterator_c = tqdm.tqdm(range(num_epochs), desc=f"Training Deep Kernel GP Combined") if verbose else range(num_epochs)
    for epoch in train_iterator_c:
        optimizer_c.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_obj.squeeze(-1))
        reg_loss = SpectralRegularizer(model, coeff=coeff, n_power_iterations=power_iterations)()
        loss = loss + reg_loss
        loss.backward()
        if verbose:
            train_iterator_c.set_postfix(loss=loss.item())
        optimizer_c.step()

        # alternating training
        optimizer_c.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_obj.squeeze(-1))
        reg_loss = SpectralRegularizer(model, coeff=coeff, n_power_iterations=power_iterations)()
        loss = reg_loss
        loss.backward()
        if verbose:
            train_iterator_c.set_postfix(loss=loss.item())
        optimizer_c.step()
    
    return mll, model

def convert_to_nn_and_base_kernel(model, x_candidates):
    '''
    Convert a deep kernel GP model to a neural network and a base kernel
    Return:
    - base_model: a RBF kernel-based GP model
    - projected_x: the projected data
    '''
    feasture_extractor = model.feature_extractor
    feasture_extractor.eval()
    projected_x = feasture_extractor(x_candidates)
    projected_x = model.scale_to_bounds(projected_x)  # Make the NN values "nice"
    base_model = RBFGPRegressionModel(feasture_extractor(model.train_inputs[0]), model.train_targets.reshape(-1,1), model.likelihood, model.mean_module, model.covar_module)
    return base_model, projected_x
    




