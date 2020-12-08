import os
import yaml
import argparse
from datetime import datetime

from baseAgent import SacdAgent
from Normalenv import Normalagent
from grid2op.Agent import BaseAgent, AgentWithConverter
from grid2op.Reward import GameplayReward, L2RPNReward, FlatReward
from grid2op.Converter import IdToAct
import grid2op

def run(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Create environments.
    env = grid2op.make("l2rpn_neurips_2020_track1_small", reward_class=L2RPNReward)
    test_env = grid2op.make("l2rpn_neurips_2020_track1_small", reward_class=L2RPNReward)


    # Specify the directory to log.
    name = args.config.split('/')[-1].rstrip('.yaml')
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join( 'logs', f'{name}-seed{args.seed}-{time}')

    # Create the agent.
    agent = SacdAgent(env=env, test_env=test_env, log_dir=log_dir, cuda=args.cuda, seed=args.seed, **config)
    agent.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('configs', 'sacd.yaml'))
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    run(args)
