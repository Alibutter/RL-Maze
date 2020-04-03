from tools.config import *
from envs.maze_env import MazeEnv
from RL_brain.q.q_learning import QL
from RL_brain.sarsa.sarsa_learning import Sarsa
from RL_brain.sarsa_lambda.sarsa_lambda import SarsaLambda
from RL_brain.dqn.dqn_learning import DQN


if __name__ == '__main__':
    env = MazeEnv()
    ql = QL(env)
    sarsa = Sarsa(env)
    sarsa_lambda = SarsaLambda(env)
    dqn = DQN(env)
    # 添加按钮
    env.add_button(5, 5, 80, 20, env.normal, env.active, env.down,
                   env.set_refresh, Strings.REFRESH, env.button_font, Color.YELLOW)
    env.add_button(5, 90, 80, 20, env.normal, env.active, env.down,
                   ql.q_learning_start, Strings.Q_LEARN, env.button_font, Color.YELLOW)
    env.add_button(5, 175, 80, 20, env.normal, env.active, env.down,
                   sarsa.sarsa_start, Strings.SARSA, env.button_font, Color.YELLOW)
    env.add_button(5, 260, 80, 20, env.normal, env.active, env.down,
                   sarsa_lambda.sarsa_lambda_start, Strings.S_LAMBDA, env.button_font, Color.YELLOW)
    env.add_button(5, 345, 80, 20, env.normal, env.active, env.down,
                   dqn.dqn_start, Strings.DQN, env.button_font, Color.YELLOW)
    while True:
        env.update_map()
