'''
This module contains the functions for Bayesian optimization
'''
import torch
import gpytorch
import numpy as np
from tqdm import tqdm
from botorch import fit_gpytorch_mll
from torch.quasirandom import SobolEngine
from botorch.acquisition import PosteriorMean
from botorch.acquisition.max_value_entropy_search import qMultiFidelityMaxValueEntropy
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.optim.optimize import optimize_acqf, optimize_acqf_discrete
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.optim.optimize import optimize_acqf_mixed
from botorch.utils.transforms import unnormalize
from src.utility import generate_initial_data, initialize_mf_model, get_project
from src import convert_to_nn_and_base_kernel, monte_carlo_entropy, rbf_kernel_variance_reduction_rate, excessive_risk_reduction_rate
from src.model import init_model_srdk
from src.acquisition import qLowerConfidenceBound, qConstrainedCI_m_UCB, qConfidenceInterval, qUpperConfidenceBound


def get_mfkg(model, problem, cost_aware_utility, project):
    '''
    Get the multi-fidelity knowledge gradient acquisition function
    '''

    curr_val_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=problem.dim,
        columns=[problem.dim-1],
        values=[1],
    )

    _, current_value = optimize_acqf(
        acq_function=curr_val_acqf,
        bounds=problem.bounds[:, :-1],
        q=1,
        num_restarts=10,
        raw_samples=1024,
        options={"batch_limit": 10, "maxiter": 200},
    )

    return qMultiFidelityKnowledgeGradient(
        model=model,
        num_fantasies=128,
        current_value=current_value, # current best
        cost_aware_utility=cost_aware_utility,
        project=project,
    )

def optimize_mfacq_and_get_observation(mfkg_acqf, fixed_features_list, problem, cost_model, batch_size, num_restarts, raw_samples):
    """Optimizes MFKG and returns a new candidate, observation, and cost."""

    # generate new candidates
    candidates, _ = optimize_acqf_mixed(
        acq_function=mfkg_acqf,
        bounds=problem.bounds,
        fixed_features_list=fixed_features_list,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        # batch_initial_conditions=X_init,
        options={"batch_limit": 5, "maxiter": 200},
    )

    # observe new values
    cost = cost_model(candidates).sum()
    new_x = candidates.detach()
    new_obj = problem(new_x).unsqueeze(-1)
    return new_x, new_obj, cost

def bo_step_kg(model, cost_aware_utility, project, fixed_features_list, problem, cost_model, batch_size, num_restarts, raw_samples):
    '''
    Perform one step of Bayesian optimization using the knowledge gradient acquisition function
    '''
    mfkg_acqf = get_mfkg(model, problem, cost_aware_utility, project)
    new_x, new_obj, cost = optimize_mfacq_and_get_observation(mfkg_acqf, fixed_features_list, problem, cost_model, batch_size, num_restarts, raw_samples)
    return new_x, new_obj, cost

def get_mfMES(model, problem, fidelities, cost_aware_utility, project, sample_n:int=40000):
    '''
    Get the multi-fidelity entropy search acquisition function
    '''
    # sample candidates
    cand_x_full, _ = generate_initial_data(problem=problem, fidelities=fidelities, n=sample_n)
    return qMultiFidelityMaxValueEntropy(model, 
                                candidate_set=cand_x_full,
                                project=project, 
                                cost_aware_utility=cost_aware_utility)

def bo_step_mes(model, cost_aware_utility, project, fidelities, fixed_features_list, problem, cost_model, batch_size, num_restarts, raw_samples, **kwargs):
    '''
    Perform one step of Bayesian optimization using the max value entropy search acquisition function
    '''
    mf_acqf = get_mfMES(model, problem, fidelities, cost_aware_utility, project, sample_n=kwargs.get('space_sample_num', 40000))
    new_x, new_obj, cost = optimize_mfacq_and_get_observation(mf_acqf, fixed_features_list, problem, cost_model, batch_size, num_restarts, raw_samples)
    return new_x, new_obj, cost

def rmfbo(problem, fidelities, config, tkwargs):
    '''
    Run the robust multi-fidelity Bayesian optimization loop
    '''
    # load configurations
    max_cholesky_size = float("inf")  # Always use Cholesky
    stopping_criteria = config.problem.stopping_criteria
    n_iter = config.problem.n_iter if stopping_criteria != 'budget' else config.problem.max_n_iter
    n_init, low_cost,\
    MIN_VALUE,\
    batch_size, target_fidelity = \
    config.problem.n_init, config.problem.low_cost,\
    config.problem.min_value,\
    config.problem.batch_size, int(config.problem.target_fidelity)

    max_opt_iter, batch_limit, = config.algorithm.max_opt_iter, config.algorithm.batch_limit,
    bo_beta, filter_beta, rate_beta, rbf_beta = config.algorithm.bo_beta, config.algorithm.filter_beta,  config.algorithm.rate_beta, config.algorithm.rbf_beta
    space_sample_num, model_sample_num, sample_train_num = config.algorithm.space_sample_num, config.algorithm.model_sample_num, config.algorithm.sample_train_num, 

    costs = fidelities + low_cost

    # Generate initial config
    sobol_eng = SobolEngine(dimension=problem.dim-1, scramble=True)


    # optimization loop
    train_x, train_obj = generate_initial_data(problem=problem, fidelities=fidelities, n=n_init)
    cumulative_cost = []
    _max_lcb = float("-inf")

    bo_iterator = tqdm(range(n_iter))
    for r_idx in bo_iterator:
        # Fit GP models for objective and constraints
        mll, model = init_model_srdk(train_x, train_obj, training_iter=max_opt_iter, verbose=False)
        x_candidates, _ = generate_initial_data(problem=problem, fidelities=fidelities, n=space_sample_num)
        target_fid_filter = x_candidates[..., -1] == target_fidelity
        while sum(target_fid_filter) == 0:
            x_candidates, _ = generate_initial_data(problem=problem, fidelities=fidelities, n=space_sample_num)
            target_fid_filter = x_candidates[..., -1] == target_fidelity
        base_model, projected_x = convert_to_nn_and_base_kernel(model, x_candidates)
        mc_f_lcb = qLowerConfidenceBound(base_model, beta=bo_beta)
        mc_f_ci = qConfidenceInterval(base_model, beta=bo_beta)

        # sample model for entropy
        # lengthscale_list = [model.covar_module.base_kernel.lengthscale.detach().clone().mean()]
        lengthscale_list = []
        for _ in range(model_sample_num):
            _, tmp_model = init_model_srdk(train_x, train_obj, training_iter=sample_train_num, lr=1e-2, power_iterations=10, verbose=False)
            tmp_lengthscale = tmp_model.covar_module.base_kernel.lengthscale.detach().clone().mean()
            lengthscale_list.append(tmp_lengthscale)
        int_lengthscale_list = torch.stack([torch.ceil(lengthscale * 1e3)/1e3 for lengthscale in lengthscale_list], dim=0)
        lengthscale_entropy = monte_carlo_entropy(int_lengthscale_list)


        # define threshold
        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
            _, _tmp_max_lcb = optimize_acqf_discrete(
                acq_function=mc_f_lcb,
                q=1,
                choices=projected_x[target_fid_filter],
                )
            _, _tmp_max_ci = optimize_acqf_discrete(
                acq_function=mc_f_ci,
                q=1,
                choices=projected_x[target_fid_filter],
                )
            
        _max_lcb = max(_max_lcb, _tmp_max_lcb)
        _tmp_max_ci = max(_tmp_max_ci, 1e-1 * torch.ones(1, **tkwargs))
        if sum(train_x[..., -1] == target_fidelity) > 0:
            _max_lcb = min(_max_lcb, train_obj[train_x[..., -1] == target_fidelity].max().item())
        model_list = [base_model]
        threshold_list = [_max_lcb]

        mc_f = qConstrainedCI_m_UCB(model_list, threshold_list, beta=bo_beta, filter_beta=filter_beta, return_UCB=True, constrained=False, sample_num=config.algorithm.sample_num)

        # optimize acquisition function
        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
            if sum(train_x[..., -1] == target_fidelity) > 0:
                center_idx = train_obj[train_x[:, -1] == target_fidelity].argmax().item()
                center = train_x[train_x[:, -1] == target_fidelity][center_idx]
                diameter = (model.covar_module.base_kernel.lengthscale / (problem.bounds[1, :-1] - problem.bounds[0, :-1])).min().item()
                x_distances = torch.linalg.norm(center - x_candidates, ord=2, dim=-1)
                # diameter_filter = x_distances < diameter * config.algorithm.diameter_scaler
                diameter_filter = x_distances < torch.quantile(x_distances, config.algorithm.distance_quantile)
                x_cand_filter = torch.logical_and(diameter_filter, target_fid_filter)
                if sum(x_cand_filter) == 0:
                    x_cand_filter = target_fid_filter
            else:
                x_cand_filter = target_fid_filter
            try:
                _choices = x_cand_filter if sum(x_cand_filter) > 0 else target_fid_filter
                f_X_next, f_X_next_idx = optimize_acqf_discrete(
                        acq_function=mc_f,
                        q=1,
                        choices=projected_x[_choices],
                        options={"batch_limit": batch_limit, "maxiter": max_opt_iter},
                    )
            except:
                _choices = target_fid_filter
                f_X_next = projected_x[_choices][torch.randint(0, sum(_choices), (1,))]
            z_next = f_X_next.clone()
            X_next = x_candidates[_choices][torch.abs(projected_x[_choices] - z_next).sum(dim=-1)==0]
            _portion = sum(_choices) / space_sample_num
        
        # optimize cost-aware acquisition function
        _t = train_x.size(0)
        _t_fids = torch.tensor([sum(train_x[:, -1] == fid) for fid in fidelities])
        rbf_rate = rbf_kernel_variance_reduction_rate(T=_t, dim=problem.dim, variance=_tmp_max_ci) * rbf_beta
        cost_aware_rbf_rate = rbf_rate / costs[-1]
        cost_aware_learning_rate = torch.sqrt(lengthscale_entropy) * torch.tensor([excessive_risk_reduction_rate(fidelity_counts=_t_fids, fidelity_choosen=i, beta=rate_beta) / costs[i] for i in range(len(fidelities))])
        if len(X_next) > batch_size:
            X_next = X_next[:batch_size]
        if cost_aware_rbf_rate > max(cost_aware_learning_rate):
            X_next = X_next.reshape(batch_size, problem.dim)
        else:
            _fid_next = np.argmax(cost_aware_learning_rate)
            _X_next = sobol_eng.draw(batch_size)
            _X_next = unnormalize(_X_next, problem.bounds[:,:-1])
            X_next = torch.cat([_X_next, torch.tensor([fidelities[_fid_next]]).repeat(batch_size, 1)], dim=-1)

        # Evaluate both the objective and constraints for the selected candidaates
        Y_next = problem(X_next)
        fid_idx = (fidelities==X_next[..., -1] ).nonzero().item()
        cost = costs[fid_idx]

        # Append data. Note that we append all data, even points that violate
        # the constraints. This is so our constraint models can learn more
        # about the constraint functions and gain confidence in where violations occur.
        train_x = torch.cat((train_x, X_next), dim=0)
        train_obj = torch.cat((train_obj, Y_next.reshape(1, -1)), dim=0)

        # Update progress bar
        cumulative_cost.append(cost)
        best_obs =  train_obj[train_x[:, -1] == target_fidelity].max().item() if sum(train_x[:, -1] == target_fidelity) > 0 else MIN_VALUE
        info = f"Max: {best_obs:.2e}"
        info += f" | Cost: {cost:.2e}"
        info += f" | new_fid: {X_next[0][-1].item():.2f}"
        info += f" | new_y: {Y_next.item():.2f}"
        info += f" | CI: {_tmp_max_ci.item():.2f}"
        info += f" | LCB: {_tmp_max_lcb.item():.2f}"
        # info += f" | Entropy: {lengthscale_entropy.item():.2f}"
        info += f"| Portion: {_portion:.2e}"
        info += f" | RBF: {cost_aware_rbf_rate.item():.2f}"
        info += f" | RRate: {max(cost_aware_learning_rate).item():.2e}"
        info += f" | Cost/Budget: {sum(cumulative_cost):.2f}/{config.problem.budget:d}"
        bo_iterator.set_postfix_str(info)
        
        # check budget
        if stopping_criteria == 'budget' and sum(cumulative_cost) > config.problem.budget:
            break
    
    return train_x, train_obj, cumulative_cost

def rmfbo_random(problem, fidelities, config, tkwargs):
    '''
    Run the robust multi-fidelity Bayesian optimization loop.
    In this version, the acquisition function is random sample within ROI.
    '''
    # load configurations
    max_cholesky_size = float("inf")  # Always use Cholesky
    stopping_criteria = config.problem.stopping_criteria
    n_iter = config.problem.n_iter if stopping_criteria != 'budget' else config.problem.max_n_iter
    n_init, low_cost,\
    MIN_VALUE,\
    batch_size, target_fidelity = \
    config.problem.n_init, config.problem.low_cost,\
    config.problem.min_value,\
    config.problem.batch_size, config.problem.target_fidelity

    max_opt_iter, batch_limit, = config.algorithm.max_opt_iter, config.algorithm.batch_limit,
    bo_beta, filter_beta, rate_beta, rbf_beta = config.algorithm.bo_beta, config.algorithm.filter_beta,  config.algorithm.rate_beta, config.algorithm.rbf_beta
    space_sample_num, model_sample_num, sample_train_num = config.algorithm.space_sample_num, config.algorithm.model_sample_num, config.algorithm.sample_train_num, 

    costs = fidelities + low_cost

    # Generate initial config
    sobol_eng = SobolEngine(dimension=problem.dim-1, scramble=True)


    # optimization loop
    train_x, train_obj = generate_initial_data(problem=problem, fidelities=fidelities, n=n_init)
    cumulative_cost = []
    _max_lcb = float("-inf")

    bo_iterator = tqdm(range(n_iter))
    for r_idx in bo_iterator:
        # Fit GP models for objective and constraints
        mll, model = init_model_srdk(train_x, train_obj, training_iter=max_opt_iter, verbose=False)
        x_candidates, _ = generate_initial_data(problem=problem, fidelities=fidelities, n=space_sample_num)
        target_fid_filter = x_candidates[..., -1] == target_fidelity
        while sum(target_fid_filter) == 0:
            x_candidates, _ = generate_initial_data(problem=problem, fidelities=fidelities, n=space_sample_num)
            target_fid_filter = x_candidates[..., -1] == target_fidelity
        base_model, projected_x = convert_to_nn_and_base_kernel(model, x_candidates)
        mc_f_ci = qConfidenceInterval(base_model, beta=bo_beta)
        mc_f_ucb = qUpperConfidenceBound(base_model, beta=bo_beta)
        mc_f_lcb = qLowerConfidenceBound(base_model, beta=bo_beta)



        # sample model for entropy
        # lengthscale_list = [model.covar_module.base_kernel.lengthscale.detach().clone().mean()]
        lengthscale_list = []
        for _ in range(model_sample_num):
            _, tmp_model = init_model_srdk(train_x, train_obj, training_iter=sample_train_num, lr=1e-2, power_iterations=10, verbose=False)
            tmp_lengthscale = tmp_model.covar_module.base_kernel.lengthscale.detach().clone().mean()
            lengthscale_list.append(tmp_lengthscale)
        int_lengthscale_list = torch.stack([torch.ceil(lengthscale * 1e3)/1e3 for lengthscale in lengthscale_list], dim=0)
        lengthscale_entropy = monte_carlo_entropy(int_lengthscale_list)

        # random sample within ROI
        diameter_filter = torch.ones(space_sample_num, dtype=torch.bool)
        if sum(train_x[..., -1] == target_fidelity) > 0:
            center_idx = train_obj[train_x[:, -1] == target_fidelity].argmax().item()
            center = train_x[train_x[:, -1] == target_fidelity][center_idx]
            x_distances = torch.linalg.norm(center - x_candidates, ord=2, dim=-1)
            diameter_filter = x_distances < torch.quantile(x_distances, config.algorithm.distance_quantile)
            x_cand_filter = torch.logical_and(diameter_filter, target_fid_filter)
            # _tmp_max_lcb = max([mc_f_lcb(model.feature_extractor(x).reshape([-1, projected_x.size(-1)])) for x in train_x[train_x[:, -1] == target_fidelity]]) 
            # x_cand_filter = torch.cat([(mc_f_ucb(model.feature_extractor(x).reshape([-1,  projected_x.size(-1)])) > _tmp_max_lcb) * torch.ones(1, dtype=bool) if x[-1] == target_fidelity else torch.zeros(1, dtype=bool) for x in x_candidates ], dim=0)
            if sum(x_cand_filter) == 0:
                x_cand_filter = target_fid_filter
        else:
            x_cand_filter = target_fid_filter
            if sum(x_cand_filter) == 0:
                x_cand_filter = target_fid_filter
        _portion = sum(x_cand_filter) / space_sample_num

        X_next = x_candidates[x_cand_filter][torch.randint(0, sum(x_cand_filter), (batch_size,))]
    
        # optimize cost-aware acquisition function
        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
            _, _tmp_max_ci = optimize_acqf_discrete(
                acq_function=mc_f_ci,
                q=1,
                choices=projected_x[target_fid_filter],
                )
        _tmp_max_ci = max(_tmp_max_ci, 1e-1 * torch.ones(1, **tkwargs))
        _t = train_x.size(0)
        _t_fids = torch.tensor([sum(train_x[:, -1] == fid) for fid in fidelities])
        rbf_rate = rbf_kernel_variance_reduction_rate(T=_t, dim=problem.dim, variance=_tmp_max_ci) * rbf_beta
        cost_aware_rbf_rate = rbf_rate / costs[-1]
        cost_aware_learning_rate = torch.sqrt(lengthscale_entropy) * torch.tensor([excessive_risk_reduction_rate(fidelity_counts=_t_fids, fidelity_choosen=i, beta=rate_beta) / costs[i] for i in range(len(fidelities))])
        if cost_aware_rbf_rate > max(cost_aware_learning_rate):
            X_next = X_next.reshape(batch_size, problem.dim)
        else:
            if sum(diameter_filter) == 0:
                X_next = x_candidates[torch.randint(0, space_sample_num, (batch_size,))] # constrain low fidelity sampling
            else:
                X_next = x_candidates[diameter_filter][torch.randint(0, sum(diameter_filter), (batch_size,))] # constrain low fidelity sampling

        # Evaluate both the objective and constraints for the selected candidaates
        Y_next = problem(X_next)
        fid_idx = (fidelities==X_next[..., -1] ).nonzero().item()
        cost = costs[fid_idx]

        # Append data. Note that we append all data, even points that violate
        # the constraints. This is so our constraint models can learn more
        # about the constraint functions and gain confidence in where violations occur.
        train_x = torch.cat((train_x, X_next), dim=0)
        train_obj = torch.cat((train_obj, Y_next.reshape(1, -1)), dim=0)

        # Update progress bar
        cumulative_cost.append(cost)
        best_obs =  train_obj[train_x[:, -1] == target_fidelity].max().item() if sum(train_x[:, -1] == target_fidelity) > 0 else MIN_VALUE
        info = f"Max: {best_obs:.2e}"
        info += f" | Cost: {cost:.2e}"
        info += f" | new_fid: {X_next[0][-1].item():.2f}"
        info += f" | new_y: {Y_next.item():.2f}"
        info += f" | CI: {_tmp_max_ci.item():.2f}"
        info += f" | Entropy: {lengthscale_entropy.item():.2f}"
        info += f" | RBF: {cost_aware_rbf_rate.item():.2f}"
        info += f" | RRate: {max(cost_aware_learning_rate).item():.2e}"
        info += f" | Valid portion: {_portion:.2f}"
        info += f" | Cost/Budget: {sum(cumulative_cost):.2f}/{config.problem.budget:d}"
        bo_iterator.set_postfix_str(info)
        
        # check budget
        if stopping_criteria == 'budget' and sum(cumulative_cost) > config.problem.budget:
            break
    
    return train_x, train_obj, cumulative_cost

def rmfbo_pseudo(problem, fidelities, config, tkwargs):
    '''
    Run the robust multi-fidelity Bayesian optimization loop adapted from https://github.com/AaltoPML/rMFBO
    Reference:
    Multi-Fidelity Bayesian Optimization with Unreliable Information Sources

    '''
    # load configurations
    max_cholesky_size = float("inf")  # Always use Cholesky
    stopping_criteria = config.problem.stopping_criteria
    n_iter = config.problem.n_iter if stopping_criteria != 'budget' else config.problem.max_n_iter
    n_init, low_cost,\
    MIN_VALUE,\
    batch_size, target_fidelity = \
    config.problem.n_init, config.problem.low_cost,\
    config.problem.min_value,\
    config.problem.batch_size, config.problem.target_fidelity

    costs = fidelities + low_cost
    fid_idx = problem.dim-1
    bounds = torch.tensor(problem.bounds, **tkwargs)
    sf_fidelities = torch.tensor([config.problem.fidelities[-1]], **tkwargs)
    sf_fixed_features_list=[{fid_idx: fid} for fid in sf_fidelities]
    target_fidelity = config.problem.target_fidelity 
    target_fidelities = {fid_idx: config.problem.target_fidelity}
    fixed_features_list=[{fid_idx: fid} for fid in fidelities]

    NEGATE = config.problem.negate
    N_INIT = config.problem.n_init
    N_ITER = config.problem.n_iter if stopping_criteria != 'budget' else config.problem.max_n_iter
    NUM_RESTARTS = config.problem.num_restarts
    RAW_SAMPLES = config.problem.raw_samples
    BATCH_SIZE = config.problem.batch_size
    SCALE = config.problem.scale
    MIN_VALUE = config.problem.min_value
    n_repeat = config.problem.n_repeat

    cost_model = AffineFidelityCostModel(fidelity_weights={fid_idx: 1.0}, fixed_cost=low_cost)
    cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)
    project = get_project(target_fidelities)

    # Generate initial config
    sobol_eng = SobolEngine(dimension=problem.dim-1, scramble=True)


    # optimization loop
    train_x, train_obj = generate_initial_data(problem=problem, fidelities=fidelities, n=n_init)
    cumulative_cost = []
    _max_lcb = float("-inf")

    bo_iterator = tqdm(range(n_iter))
    for r_idx in bo_iterator:
        # multi-fidelity subroutine
        mll, model = initialize_mf_model(train_x, train_obj, data_fidelity=fid_idx)
        fit_gpytorch_mll(mll)
        if config.algorithm.acq_name == 'KG':
            new_x, new_obj, cost = bo_step_kg(model, cost_aware_utility, project, fixed_features_list, problem, cost_model, BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES)
        elif config.algorithm.acq_name == 'MES':
            new_x, new_obj, cost = bo_step_mes(model, cost_aware_utility, project, fidelities, fixed_features_list, problem, cost_model, BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES, space_sample_num=config.algorithm.space_sample_num)
        elif config.algorithm.acq_name == 'RAND':
            new_x, new_obj = generate_initial_data(problem=problem, fidelities=fidelities, n=1)
            new_fid_val = new_x[..., -1].item()
            cost = low_cost + new_fid_val
        else:                    
            raise NotImplementedError
        
        # check if going to SF subroutine
        cond_var = model.posterior(new_x).variance <= (0.5 ** 2)
        cond_is = qMultiFidelityMaxValueEntropy(model=model, candidate_set=new_x)(X=new_x) >= 0.5
        if not (cond_var and cond_is):
            if config.algorithm.acq_name == 'KG':
                new_x, new_obj, cost = bo_step_kg(model, cost_aware_utility, project, sf_fixed_features_list, problem, cost_model, BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES)
            elif config.algorithm.acq_name == 'MES':
                new_x, new_obj, cost = bo_step_mes(model, cost_aware_utility, project, sf_fidelities, sf_fixed_features_list, problem, cost_model, BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES, space_sample_num=config.algorithm.space_sample_num)
            elif config.algorithm.acq_name == 'RAND':
                new_x, new_obj = generate_initial_data(problem=problem, fidelities=sf_fidelities, n=1)
                new_fid_val = new_x[..., -1].item()
                cost = low_cost + new_fid_val
            else:                    
                raise NotImplementedError

        train_x = torch.cat([train_x, new_x])
        train_obj = torch.cat([train_obj, new_obj])
        best_obs =  train_obj[train_x[:, -1] == target_fidelity].max().item() if sum(train_x[:, -1] == target_fidelity) > 0 else MIN_VALUE
        cumulative_cost.append(cost)
        info = f"Max: {best_obs:.2e}"
        info += f" | Cost: {cost:.2e}"
        info += f" | new_fid: {new_x[0][-1].item():.2f}"
        info += f" | new_y: {new_obj.item():.2f}"
        info += f" | Cost/Budget: {sum(cumulative_cost):.2f}/{config.problem.budget:d}"
        bo_iterator.set_postfix_str(info)

        # check budget
        if stopping_criteria == 'budget' and sum(cumulative_cost) > config.problem.budget:
            break

    
    return train_x, train_obj, cumulative_cost

