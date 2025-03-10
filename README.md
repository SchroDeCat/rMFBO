# Robust Multi-fidelity BO with Deep Kernel and Partition

## Environment Setup

To set up the environment, the code has been tested on a Macbook Pro 2021 (Mac OS 14.6.1 (23G93)) with 16GB RAM and an M1 Chip.

Install the required dependencies listed in **requirements.txt** using the following command:

```shell
pip install -r requirements.txt
```

## Project Structure

The main implementation is located in the **./src** directory, while the hydra configuration files are in the **./config** directory.

```plaintext
rMFBO/
├── conf/
│   ├── __init__.py
│   ├── algorithm/
│   │   ├── kg.yaml
│   │   ├── mes.yaml
│   │   ├── random.yaml
│   │   ├── rmfbo.yaml
│   │   ├── rmfbo_pseudo.yaml
│   │   └── rmfbo_random.yaml
│   ├── experiment.yaml
│   └── problem/
│       ├── iaml_rpart.yaml
│       ├── iaml_xgboost.yaml
│       ├── lcbench.yaml
│       ├── protein.yaml
│       ├── rastrigin.yaml
│       └── rastrigin20d.yaml
├── main.py
├── README.md
├── requirements.txt
└── src/
    ├── __init__.py
    ├── acquisition.py
    ├── model.py
    ├── optim.py
    ├── test_functions.py
    └── utility.py
```

The `main.py` script serves as the entry point for benchmarking. For example, to run the script multiple times on the Rastrigin 1D problem with specified algorithms, use the following command:

```shell
python main.py -m problem=rastrigin algorithm=rmfbo,rmfbo_pseudo,mes,kg
```


## Citation

If you wish to cite the project, please consider the following

```bash
@inproceedings{zhang2024robust,
  title={Robust Multi-fidelity Bayesian Optimization with Deep Kernel and Partition},
  author={Zhang, Fengxue and Desautels, Thomas and Chen, Yuxin},
  booktitle={28th International Conference on Artificial Intelligence and Statistics}
}
```