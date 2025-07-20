import warnings
from pathlib import Path

import pandas as pd

import src.vizualization as viz
from src.environment import setup_environment
from src.params import params
from src.q_learning import EpsilonGreedy, QLearning
from src.run_env import run_env

warnings.filterwarnings("ignore")


def run():
    results_dir = Path("results")
    plots_dir = results_dir / "plots"

    results_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    _map_sizes = [4, 7, 9, 11]
    res_all = pd.DataFrame()
    st_all = pd.DataFrame()

    for map_size in _map_sizes:
        params.map_size = map_size

        env = setup_environment(name_prefix=f"{map_size}x{map_size}-eval")

        params.action_size = env.action_space.n
        params.state_size = env.observation_space.n
        env.action_space.seed(params.seed)

        learner = QLearning(
            learning_rate=params.learning_rate,
            gamma=params.gamma,
            state_size=params.state_size,
            action_size=params.action_size,
        )
        explorer = EpsilonGreedy(epsilon=params.epsilon)

        print(f"Map size: {map_size}x{map_size}")
        rewards, steps, episodes, qtables, all_states, all_actions = run_env(
            learner=learner, explorer=explorer, env=env
        )

        res, st = viz.postprocess(episodes, params, rewards, steps, map_size)
        res_all = pd.concat([res_all, res])
        st_all = pd.concat([st_all, st])
        qtable = qtables.mean(axis=0)

        viz.plot_states_actions_distribution(
            states=all_states,
            actions=all_actions,
            save_path=str(
                plots_dir / f"states_actions_distribution_{map_size}x{map_size}.png"
            ),
        )
        viz.plot_q_values_map(
            qtable=qtable,
            env=env,
            map_size=map_size,
            save_path=str(plots_dir / f"q_values_map_{map_size}x{map_size}.png"),
        )

        env.close()

    viz.plot_steps_and_rewards(
        res_all, st_all, save_path=str(plots_dir / "training_progress_all_sizes.png")
    )


if __name__ == "__main__":
    run()
