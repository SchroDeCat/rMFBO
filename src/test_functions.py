'''
Personalized toy example for testing the multi-fidelity optimization algorithms.
'''
from __future__ import annotations

import math
from typing import Optional

import torch
from botorch.test_functions.synthetic import SyntheticTestFunction
from botorch.utils.transforms import unnormalize
from torch import Tensor
import numpy as np


class AugmentedRastrigin(SyntheticTestFunction):
    r"""  
    Augmented Rastrigin test function for multi-fidelity optimization.

    1-dimensional function with domain `[-5, 5] * [0,1]`, where
    the last dimension of is the fidelity parameter:

        B(x) = -(x**2 - 10 * cos(2 * pi * x) + 10)
    low fids:
        y1 = y + np.random.normal(1 * np.abs(x))
        y2 = y + np.random.normal(2 * x**2, 8)
        y3 = y + np.random.normal(-.2 * x**4, 8)

    B_min ~= 40.2 which are on the sides
    B_max = 0 which is in the middle
    Discrete fidelities:
        fid = [0.1, 0.5, 0.75, 1]
    """

    dim = 2 # last is fidelity
    _bounds = [(-10.0, 10.0), (0, 1)]
    # _optimal_value = 40.2
    _optimal_value = 0
    _optimizers = [  # there are two
        # (-4.49), (4.49)
        (0.0, 1),
    ]
    _scale = 1.0

    def evaluate_true(self, X: Tensor) -> Tensor:
        single_fid_x = X[..., :-1].squeeze()
        base = (single_fid_x**2 - 10 * torch.cos(2 * math.pi * single_fid_x) + 10)
        fid = X[..., -1]
        fid_correction_mu = ((1 - fid) > 1e-10) * (2 ** ((fid) <= 0.5)) * ((-.1) ** ((fid) <= 0.25))
        fid_correction_mu = fid_correction_mu * (torch.abs(single_fid_x) **((1-fid)/0.25))
        fidelity_correction = torch.normal(mean=fid_correction_mu, std=8) * ((1 - fid) > 1e-10)
        return (- base - fidelity_correction) * self._scale
        # return -base

class FixedProtein(SyntheticTestFunction):
    r'''
    Fixed Protein test function for multi-fidelity optimization.
    '''
    dim = 87
    protein_data = torch.load("data/fixed_protein.pt").to(torch.float64)

    def __init__(self, negate: Optional[bool] = False) -> None:
        self._bounds = [(self.protein_data[:,d].min(), self.protein_data[:,d].max()+1e-6) for d in range(self.dim)]
        self.candidates = self.protein_data[:,:-1]
        self.objectives = self.protein_data[:,-1].reshape(-1, 1)
        self._optimal_value = self.protein_data[self.protein_data[:,-2]==2,-1].min()
        # self.protein_data = self.protein_data[np.random.choice(1000, self.protein_data.size(0))]
        super().__init__(negate=negate)
        # self.protein_data = self.protein_data.to(**self._tkwargs)

    def evaluate_true(self, X: Tensor) -> Tensor:
        # find closest protein
        indices = []
        # base = self.protein_data[np.random.choice(1000, self.protein_data.size(0)),:-1] 
        base = self.protein_data[:,:-1] 
        for x in X:
            # dist = torch.linalg.norm(self.protein_data[...,:-1] - x, ord=2, dim=-1)
            dist = torch.linalg.norm(base-x, ord=1, dim=-1)
            idx = torch.argmin(dist)
            indices.append(idx)
        return self.protein_data[indices, -1]

class AugmentedRastrigin20D(SyntheticTestFunction):
    r"""  
    Augmented Rastrigin test function for multi-fidelity optimization.

    1-dimensional function with domain `[-5, 5]^dim * [0,1]`, where
    the last dimension of is the fidelity parameter:

        B(x) = -(x**2 - 10 * cos(2 * pi * x) + 10)
    low fids:
        y1 = y + np.random.normal(1 * np.abs(x))
        y2 = y + np.random.normal(2 * x**2, 8)
        y3 = y + np.random.normal(-.2 * x**4, 8)

    B_max = 0 which is in the middle
    Discrete fidelities:
        fid = [0.1, 0.5, 0.75, 1]
    """

    dim = 21 # last is fidelity
    _bounds = [(-5.0, 5.0) for _ in range(dim-1)]
    _bounds.append((0, 1))
    # _optimal_value = 40.2
    _optimal_value = 0
    _optimizers = [  # there are two
        # (-4.49), (4.49)
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1),
    ]
    _scale = 1.0

    def evaluate_true(self, X: Tensor) -> Tensor:
        single_fid_x = X[..., :-1].squeeze()
        base = (torch.linalg.norm(single_fid_x, ord=2)**2 - 10 * torch.cos(2 * math.pi * single_fid_x.max()) + 10)
        fid = X[..., -1]
        fid_correction_mu = ((1 - fid) > 1e-10) * (2 ** ((fid) <= 0.5)) * ((-.1) ** ((fid) <= 0.25))
        fid_correction_mu = fid_correction_mu * (torch.linalg.norm(single_fid_x) **((1-fid)/0.25))
        fidelity_correction = torch.normal(mean=fid_correction_mu, std=8) * ((1 - fid) > 1e-10)
        return (- base - fidelity_correction) * self._scale
        # return -base

class YahooGYM(SyntheticTestFunction):
    '''
    Paper found in https://arxiv.org/abs/2109.03670
    Github: https://github.com/slds-lmu/yahpo_gym
    '''

    pass

class LCBench(YahooGYM):
    '''
    Part of YahooGYM, 7D Numeric, 34 instnces, 6 objectives, Fidelity defined by the number of epochs
    - instances: default 3945
        ['3945', '7593', '34539', '126025', '126026', '126029', '146212', '167104', '167149', 
        '167152', '167161', '167168', '167181', '167184', '167185', '167190', '167200', '167201', 
        '168329', '168330', '168331', '168335', '168868', '168908', '168910', '189354', '189862', 
        '189865', '189866', '189873', '189905', '189906', '189908', '189909']
    - objectives:
        [{'time', 'val_accuracy', 'val_cross_entropy', 'val_balanced_accuracy', 'test_cross_entropy', 'test_balanced_accuracy'}]
        We pick test_balanced_accuracy as the objective to be optimized (maximize)
    - citation:
        'Zimmer, L. (2020). data_2k_lw.zip. figshare. Dataset. 
        https://doi.org/10.6084/m9.figshare.11662422.v1, Apache License, Version 2.0.']
    - discrete fidelities (1-52), default 26
        [1, 8, 16, 32]
    - inputs:
        batch_size, Type: UniformInteger, Range: [16, 512], Default: 91, on log-scale
        learning_rate, Type: UniformFloat, Range: [0.00010000000000000009, 0.10000000000000002], Default: 0.0031622777, on log-scale
        max_dropout, Type: UniformFloat, Range: [0.0, 1.0], Default: 0.5
        max_units, Type: UniformInteger, Range: [64, 1024], Default: 256, on log-scale
        momentum, Type: UniformFloat, Range: [0.1, 0.99], Default: 0.545
        num_layers, Type: UniformInteger, Range: [1, 5], Default: 3
        weight_decay, Type: UniformFloat, Range: [1e-05, 0.1], Default: 0.050005
        epoch, Type: UniformInteger, Range: [1, 52], Default: 26
    - reference:
        https://www.tnt.uni-hannover.de/papers/data/1459/2020_arXiV_Auto_PyTorch.pdf
    '''
    dim = 8
    _bounds = [
        (16, 512),
        (0.0001001, 0.1),
        (0.0, 1.0),
        (64, 1024),
        (0.1, 0.99),
        (1, 5),
        (0.0000101, 0.1),
        (1, 52)
    ]

    def __init__(self, negate: Optional[bool] = False, instance: str = '3945') -> None:
        from yahpo_gym import local_config
        from yahpo_gym import benchmark_set
        local_config.init_config()
        local_config.set_data_path("./data/yahpo_data")
        self.local_config = local_config
        self.bench = benchmark_set.BenchmarkSet("lcbench")
        self.input_keys = ['batch_size', 'learning_rate', 'max_dropout', 'max_units', 'momentum', 'num_layers', 'weight_decay', 'epoch']
        self.int_keys = ['batch_size', 'max_units', 'num_layers', 'epoch']
        assert instance in self.bench.instances
        self.bench.set_instance(instance)
        self.instance_id = instance
        super().__init__(negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        configs = []
        for x in X:
            config = {k: v for k, v in zip(self.input_keys, x) }
            for k in config.keys():
                if k in self.int_keys:
                    config[k] = np.int64(config[k])
                else:
                    config[k] = np.float64(config[k])
            config['OpenML_task_id'] = self.instance_id
            configs.append(config)

        outputs = self.bench.objective_function(configs)
        objectives = [inst['test_balanced_accuracy'] for inst in outputs]
        return torch.tensor(objectives, dtype=torch.float64)

class iaml_rpart(YahooGYM):
    '''
    Part of YahooGYM, 4D Numeric, 4 instances, 12 objectives, Fidelity defined by fraction of training data
    - instances: default '1489'
        ['40981', '41146', '1489', '1067']
    - objectives:
        ['mmce', 'f1', 'auc', 'logloss', 'ramtrain', 'rammodel', 'rampredict', 'timetrain', 'timepredict', 'mec', 'ias', 'nf']
        We pick auc as the objective to be optimized (maximize)
    - discrete trainsize (0-1), default 0.525
    - inputs:
            task_id, Type: Constant, Value: 1489
            cp, Type: UniformFloat, Range: [0.00010000000000000009, 1.0], Default: 0.01, on log-scale
            maxdepth, Type: UniformInteger, Range: [1, 30], Default: 16
            minbucket, Type: UniformInteger, Range: [1, 100], Default: 50
            minsplit, Type: UniformInteger, Range: [1, 100], Default: 50
            trainsize, Type: UniformFloat, Range: [0.03, 1.0], Default: 0.525
    '''
    dim = 5
    _bounds = [(0.001, 1.0), (1, 30), (1, 100), (1, 100), (0.03, 1.0)] # the last is fraction (fidelity)

    def __init__(self, negate: Optional[bool] = False, instance: str = '1489') -> None:
        from yahpo_gym import local_config
        from yahpo_gym import benchmark_set
        local_config.init_config()
        local_config.set_data_path("./data/yahpo_data")
        self.local_config = local_config
        self.bench =  benchmark_set.BenchmarkSet("iaml_rpart")
        self.input_keys = ['cp', 'maxdepth', 'minbucket', 'minsplit', 'trainsize']
        self.int_keys = ['maxdepth', 'minbucket', 'minsplit']
        assert instance in self.bench.instances
        self.bench.set_instance(instance)
        self.instance_id = instance
        super().__init__(negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        configs = []
        for x in X:
            config = {k: v for k, v in zip(self.input_keys, x) }
            for k in config.keys():
                if k in self.int_keys:
                    config[k] = np.int64(config[k])
                else:
                    config[k] = np.float64(config[k])
            config['task_id'] = self.instance_id
            configs.append(config)

        outputs = self.bench.objective_function(configs)
        objectives = [inst['auc'] for inst in outputs]
        return torch.tensor(objectives, dtype=torch.float64)

class iaml_xgboost(YahooGYM):
    '''
    Part of YahooGYM, 13D Numeric, 4 instnces, 12 objectives, Fidelity defined by the fraction of training data
    - instances: default '1489'
        ['40981', '41146', '1489', '1067']
    - objectives:
        ['mmce', 'f1', 'auc', 'logloss', 'ramtrain', 'rammodel', 'rampredict', 'timetrain', 'timepredict', 'mec', 'ias', 'nf']
        We pick auc as the objective to be optimized (maximize)
    - discrete trainsize (0-1), default 0.525
    - inputs:
        task_id, Type: Constant, Value: 1489
        booster, Type: Categorical, Choices: {gblinear, gbtree, dart}, Default: dart (to allow all hyperparameters to be active)

        alpha, Type: UniformFloat, Range: [0.00010000000000000009, 999.9999999999998], Default: 0.316227766, on log-scale
        colsample_bylevel, Type: UniformFloat, Range: [0.01, 1.0], Default: 0.505
        colsample_bytree, Type: UniformFloat, Range: [0.01, 1.0], Default: 0.505
        eta, Type: UniformFloat, Range: [0.00010000000000000009, 1.0], Default: 0.01, on log-scale
        gamma, Type: UniformFloat, Range: [0.00010000000000000009, 6.999999999999999], Default: 0.0264575131, on log-scale
        lambda, Type: UniformFloat, Range: [0.00010000000000000009, 999.9999999999998], Default: 0.316227766, on log-scale
        max_depth, Type: UniformInteger, Range: [1, 15], Default: 8
        min_child_weight, Type: UniformFloat, Range: [2.718281828459045, 149.99999999999997], Default: 20.1926292064, on log-scale
        nrounds, Type: UniformInteger, Range: [3, 2000], Default: 77, on log-scale
        rate_drop, Type: UniformFloat, Range: [0.0, 1.0], Default: 0.5
        skip_drop, Type: UniformFloat, Range: [0.0, 1.0], Default: 0.5
        subsample, Type: UniformFloat, Range: [0.1, 1.0], Default: 0.55
        trainsize, Type: UniformFloat, Range: [0.03, 1.0], Default: 0.525
    - conditions:
        colsample_bylevel | booster in {'dart', 'gbtree'}
        colsample_bytree | booster in {'dart', 'gbtree'}
        eta | booster in {'dart', 'gbtree'}
        gamma | booster in {'dart', 'gbtree'}
        max_depth | booster in {'dart', 'gbtree'}
        min_child_weight | booster in {'dart', 'gbtree'}
        rate_drop | booster == 'dart'
        skip_drop | booster == 'dart'
    '''
    dim = 13
    _bounds = [
            (0.0001001, 999.9999999999998),
            (0.01001, 1.0),
            (0.01001, 1.0),
            (0.0001001, 1.0),
            (0.0001001, 6.999999999999999),
            (0.0001001, 999.9999999999998),
            (1.0, 15.0),
            (2.7183, 149.99999999999997),
            (3.0, 2000.0),
            (0.0, 1.0),
            (0.0, 1.0),
            (0.1001, 1.0),
            (0.03, 1.0)
        ]
     # the last is fraction (fidelity)

    def __init__(self, negate: Optional[bool] = False, instance: str = '1489', booster:str = 'dart') -> None:
        from yahpo_gym import local_config
        from yahpo_gym import benchmark_set
        local_config.init_config()
        local_config.set_data_path("./data/yahpo_data")
        self.local_config = local_config
        self.bench =  benchmark_set.BenchmarkSet("iaml_xgboost")
        self.booster = booster
        self.input_keys = ['alpha','colsample_bylevel', 'colsample_bytree', 'eta', 'gamma', 'lambda', 
                           'max_depth', 'min_child_weight', 
                           'nrounds', 'rate_drop', 'skip_drop', 'subsample', 'trainsize']
        self.int_keys = ['max_depth', 'nrounds']
        assert instance in self.bench.instances
        self.bench.set_instance(instance)
        self.instance_id = instance
        super().__init__(negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        configs = []
        for x in X:
            config = {k: v for k, v in zip(self.input_keys, x) }
            for k in config.keys():
                if k in self.int_keys:
                    config[k] = np.int64(config[k])
                else:
                    config[k] = np.float64(config[k])
            config['task_id'] = self.instance_id
            config['booster'] = self.booster
            configs.append(config)

        outputs = self.bench.objective_function(configs)
        objectives = [inst['auc'] for inst in outputs]
        return torch.tensor(objectives, dtype=torch.float64)

class HPO_Benchmark(SyntheticTestFunction):
    '''
    Paper found in https://arxiv.org/abs/2109.06716
    Github: https://github.com/automl/HPOBench
    '''
    pass

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    problem = AugmentedRastrigin(negate=False).to(**tkwargs)
    fidelities = torch.tensor([0, 0.5, 0.75, 1.0], **tkwargs)
    fidelity_color = ['red', 'green', 'blue', 'black']
    n = 1000

    train_x = unnormalize(torch.rand(n, 1, **tkwargs).reshape(n, 1), bounds=problem.bounds[:,:-1])
    train_f = fidelities[torch.randint(4, (n, 1))]
    train_x_full = torch.cat((train_x, train_f), dim=1)
    train_obj = problem(train_x_full).unsqueeze(-1)  # add output dimension

    plt.figure()
    for x, y in zip(train_x_full, train_obj):
        if x[-1] == 0:
            label = 0
        elif x[-1] == 0.5:
            label = 1
        elif x[-1] == 0.75:
            label = 2
        else:
            label = 3
        plt.scatter(x[0].cpu().numpy(), y.cpu().numpy(), color=fidelity_color[label], s=5,)


    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'One-Dimensional Rastrigin with Different Fidelities Max {train_obj[train_f==1].max(dim=0).values:.2e}')
    plt.grid(True)
    plt.legend()
    # plt.show()
    plt.savefig(f'rastrigin_fidelity_min{train_obj[train_f==1].min():.2e}_max{train_obj[train_f==1].max():.2e}.png')
