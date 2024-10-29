import os
import torch
import hydra
import datetime
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from botorch.test_functions.multi_fidelity import AugmentedHartmann
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility

from src.utility import generate_initial_data, initialize_mf_model, get_project
from src.test_functions import AugmentedRastrigin, AugmentedRastrigin20D, FixedProtein, LCBench, iaml_rpart, iaml_xgboost
from src.opt import bo_step_kg, bo_step_mes, rmfbo, rmfbo_random, rmfbo_pseudo
from src.model import init_model_srdk

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
torch.set_printoptions(precision=3, sci_mode=False)

@hydra.main(version_base=None, config_path="conf", config_name="experiment")
def experiment(config: DictConfig):
    # load config
    stopping_criteria = config.problem.stopping_criteria
    SF = config.problem.SF if hasattr(config.problem, 'SF') else False
    NEGATE = config.problem.negate
    N_INIT = config.problem.n_init
    N_ITER = config.problem.n_iter if stopping_criteria != 'budget' else config.problem.max_n_iter
    NUM_RESTARTS = config.problem.num_restarts
    RAW_SAMPLES = config.problem.raw_samples
    BATCH_SIZE = config.problem.batch_size
    SCALE = config.problem.scale
    MIN_VALUE = config.problem.min_value
    n_repeat = config.problem.n_repeat
    dk = False
    max_opt_iter = 50
    if hasattr(config.algorithm, 'dk'):
        dk = config.algorithm.dk
    if hasattr(config.algorithm, 'max_opt_iter'):
        max_opt_iter = config.algorithm.max_opt_iter


    # init problem
    if config.problem.name == 'AugmentedRastrigin':
        problem = AugmentedRastrigin(negate=NEGATE).to(**tkwargs)
        problem.scale = SCALE
    elif config.problem.name == 'AugmentedRastrigin20D':
        problem = AugmentedRastrigin20D( negate=NEGATE).to(**tkwargs)
        problem.scale = SCALE
    elif config.problem.name == 'AugmentedHartmann':
        problem = AugmentedHartmann(negate=NEGATE).to(**tkwargs)
        problem.scale = SCALE
    elif config.problem.name == 'ProteinFixed':
        problem = FixedProtein(negate=NEGATE).to(**tkwargs)
        problem.scale = SCALE
    elif config.problem.name == 'LCBench':
        problem = LCBench(negate=NEGATE).to(**tkwargs)
        problem.scale = SCALE
    elif config.problem.name == 'RPart':
        problem = iaml_rpart(negate=NEGATE).to(**tkwargs)
        problem.scale = SCALE
    elif config.problem.name == 'XGBoost':
        problem = iaml_xgboost(negate=NEGATE).to(**tkwargs)
        problem.scale = SCALE

    else:
        raise NotImplementedError(f"Problem {config.problem.name} not implemented")

    low_cost = config.problem.low_cost
    fid_idx = problem.dim-1
    fidelities = torch.tensor(config.problem.fidelities if not SF else [config.problem.fidelities[-1]], **tkwargs)
    # fidelities = torch.tensor([0.1, 1.0], **tkwargs)
    bounds = torch.tensor(problem.bounds, **tkwargs)
    target_fidelity = config.problem.target_fidelity 
    target_fidelities = {fid_idx: config.problem.target_fidelity}
    fixed_features_list=[{fid_idx: fid} for fid in fidelities]

    cost_model = AffineFidelityCostModel(fidelity_weights={fid_idx: 1.0}, fixed_cost=low_cost)
    cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)
    project = get_project(target_fidelities)

    record = np.zeros((n_repeat, config.problem.max_n_iter, 2)) # tuple is (max_reward, cost)

    # repeated runs
    repeat_iterator = tqdm(range(n_repeat))
    for repeat_idx in repeat_iterator: 
        # optimization loop
        train_x, train_obj = generate_initial_data(problem=problem, fidelities=fidelities, n=N_INIT)
        cumulative_cost = []
        opt_iterator = tqdm(range(N_ITER))
        best_obs =  train_obj[train_x[:, -1] == target_fidelity].max().item() if sum(train_x[:, -1] == target_fidelity) > 0 else MIN_VALUE
        if config.algorithm.name == 'RMFBO': 
            # RMFBO single run
            train_x, train_obj, cumulative_cost = rmfbo(problem, fidelities, config, tkwargs)
        elif config.algorithm.name == 'RMFBO-RANDOM':
            train_x, train_obj, cumulative_cost = rmfbo_random(problem, fidelities, config, tkwargs)
        elif config.algorithm.name == 'RMFBO-PSEUDO':
            train_x, train_obj, cumulative_cost = rmfbo_pseudo(problem, fidelities, config, tkwargs)
        else:                                
            # Other single runs
            for i in opt_iterator:
                if dk:
                    mll, model = init_model_srdk(train_x, train_obj, training_iter=max_opt_iter, verbose=False)
                else:
                    mll, model = initialize_mf_model(train_x, train_obj, data_fidelity=fid_idx)
                fit_gpytorch_mll(mll)
                if config.algorithm.name == 'KG':
                    new_x, new_obj, cost = bo_step_kg(model, cost_aware_utility, project, fixed_features_list, problem, cost_model, BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES)
                elif config.algorithm.name == 'MES':
                    new_x, new_obj, cost = bo_step_mes(model, cost_aware_utility, project, fidelities, fixed_features_list, problem, cost_model, BATCH_SIZE, NUM_RESTARTS, RAW_SAMPLES, space_sample_num=config.algorithm.space_sample_num)
                elif config.algorithm.name == 'RAND':
                    new_x, new_obj = generate_initial_data(problem=problem, fidelities=fidelities, n=1)
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
                opt_iterator.set_postfix_str(info)

                # check budget
                if stopping_criteria == 'budget' and sum(cumulative_cost) > config.problem.budget:
                    break
    
        # print the maximum reward on the target fidelity
        cumulative_cost = torch.cumsum(torch.tensor(cumulative_cost), dim=0)
        target_fidelity_idx = train_x[:, -1] == target_fidelity
        max_target_fidelity = train_obj[target_fidelity_idx].max() if sum(target_fidelity_idx) > 0 else MIN_VALUE
        maximum_reward = train_obj.clone().detach()
        maximum_reward = torch.where((train_x[:, -1] == target_fidelity).unsqueeze(-1), maximum_reward, (-float('inf') if NEGATE else MIN_VALUE) * torch.ones_like(maximum_reward) * SCALE)
        maximum_reward = torch.cummax(maximum_reward, dim=0).values
        record_length =  train_x.size(0) - N_INIT
        record[repeat_idx, :record_length] = torch.cat([maximum_reward[-record_length:].reshape(record_length, 1), cumulative_cost.reshape(record_length, 1)], dim=-1).numpy()


    # store results
    __file_name = f"{config.problem.name}{'_NEG' if NEGATE else ''}_{config.algorithm.name}{'-'+config.algorithm.acq_name if config.algorithm.name == 'RMFBO-PSEUDO' else ''}"
    __file_name += f"{'-DK' if dk else ''}"
    __file_name += f"_R{n_repeat}_NI{N_INIT}_C{low_cost}"
    if stopping_criteria == 'budget':
        __file_name += f"_B{config.problem.budget}"
    else:
        __file_name += f"_NI{N_ITER}"
    if SF:
        __file_name += "_SF"
        
    
    if config.problem.plot_results: # plot last runs results
        # plot the objective
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(list(range(train_x.size(0))), maximum_reward)
        plt.xlabel('Iteration')
        plt.ylabel('f(x)')
        plt.title('Rewards')
        plt.grid(True)
        # plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(list(range(cumulative_cost.size(0))), cumulative_cost)
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('Cumulative Cost')
        plt.grid(True)
        plt.plot()
        plt.savefig(f"./res/figures/test_{__file_name}.png")
    
    hydra_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    np.save(f"{hydra_dir}/{__file_name}.npy", record)
    
    # time stamp
    now = datetime.datetime.now()
    file_name = f"./res/records/{__file_name}_{now.strftime('%Y-%m-%d %H:%M:%S')}.npy"
    np.save(f"{file_name}", record)
    print(f"Results saved to: {file_name}")


if __name__ == '__main__':
    experiment()






    

    
    
    
