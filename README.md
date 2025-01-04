
MSc in Artificial Intelligence, University of Bath
Dissertation: A Comparative Analysis of Multi-Layer Perceptron Configurations in Actor-Critic Reinforcement Learning for Portfolio Optimisation

The codebase is based on the following work:
https://github.com/AI4Finance-Foundation/FinRL-Trading/tree/master/old_repo_ensemble_strategy

Corresponding research paper:
Yang, H., Liu, X.Y., Zhong, S. and Walid, A., 2020, October. Deep reinforcement learning for automated stock trading: An ensemble strategy. In Proceedings of the first ACM international conference on AI in finance (pp. 1-8).

Please consult with the LICENSE file before use


INSTRUCTIONS:
-------------
This repository contains the code required to reproduce the results and visualisations presented in the dissertation
Please consult with the `requirements.txt` file for the necessary Python packages to be installed; note that it needs to run on Python 3.6 or earlier
It comprises three key components: a script for running single model simulations, another for ensemble simulations, and a Jupyter notebook for processing outputs and generating figures and tables

1. Running a Single Model
To run a single model:
- Open the `run_single_model.py` file
- Modify the input parameters for the `run_model` function as needed
- Ensure that the `strategy_name` matches an existing strategy in `strategies.py`; to add or modify strategies, refer to the `strategies.py` file
- Execute the script

Example configuration:
if __name__ == "__main__":
    run_model(
        strategy_name='DDPG_MLP_default_relu_TL1000',
        number_of_runs=30,
        timesteps=3_000,
    )

2. Running an Ensemble
For ensemble simulations:
- Open the `new_run_DRL.py` file
- Modify the input parameters for the `strategy_names_and_timesteps` function as needed
- Ensure that the `strategy_name` matches an existing strategy in `strategies.py`; to add or modify strategies, refer to the `strategies.py` file
- Note: The dissertation uses three strategies simultaneously, but the code supports more
        The `ensemble_strategy` must be either `'default'` or `'non-greedy'`
- Execute the script

Example configuration:
if __name__ == "__main__":
    run_model(
        strategy_names_and_timesteps={
            'A2C_MLP_default_leaky_relu_TL140': 20_000,
            'DDPG_MLP_wider_leaky_relu_TL140': 3_000,
            'PPO_MLP_deeper_leaky_relu_TL140': 20_000,
        },
        number_of_runs=30,
        ensemble_strategy='default',
    )

3. Generating Outputs and Visualisations
To obtain run statistics and generate all supporting tables and visualisations:
- Open the `Dissertation outputs.ipynb` notebook
- Execute the cells one by one to process the outputs and reproduce the dissertation's figures and tables