You can run this tool by following the steps below:

1.First of all, the python environment we use is 3.10.14.

2.You need to install some packages for using it.
```python
pip install -r requirements.txt
```

3.Our tool relies on the Gurobi solver, for which you need to obtain a license. 
You can find it at: https://www.gurobi.com/solutions/gurobi-optimizer/

4.When you have all the environment ready, you can run the tool.

Let's take C1 as an example to illustrate its use.

You can find the following code in $./benchmarks/C1.py$:

```python
def main():
    env_name = 'C1'
    example = get_examples_by_name(env_name)
    load = False
    begin = timeit.default_timer()
    opts = {
        'example': example,
        'epsilon_step': 0.1,
        'num_episodes': 30,
        'epsilon': 0.7,
        'buffer_size': 10000,
        'batch_size': 500,
        'unit': 64,
        'lr': 1e-4,
        'device': torch.device('cuda:0')
    }
    config = ProofConfig(**opts)

    env = Env(example)

    agent = DQN(config, load=load, double_dqn=True)

    if not load:
        num = 100
        train_off_policy_agent(env, agent, config, num=num)
        end = timeit.default_timer()
        print(f'Total time: {end - begin}s')
    else:
        reappear(agent, env)
```

<1>.If you want to retrain a model, you need to set the parameter $load=False$ and adjust some hyperparameters as you wish. Then run it.

<2>If you want to reproduce our results with our model, you need to set the parameter load=True and then run it.
