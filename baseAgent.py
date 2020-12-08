import os
import numpy as np
import torch
from torch.optim import Adam

from model import TwinnedQNetwork, CateoricalPolicy
from utils import disable_gradients


from torch.utils.tensorboard import SummaryWriter

from memory import LazyMultiStepMemory
from utils import update_params, RunningMeanStats
from Normalenv import Normalagent

from grid2op.Reward import GameplayReward, L2RPNReward




class SacdAgent():
    def __init__(self, env, test_env, log_dir, num_steps=10000, batch_size=128,
                 lr=0.0003, memory_size=1000000, gamma=0.99, multi_step=1,
                 target_entropy_ratio=0.98, start_steps=2000,
                 update_interval=4, target_update_interval=8000,num_eval_steps=100000,
                 max_episode_steps=10000, log_interval=10, eval_interval=1000,
                 cuda=True, seed=0):
        super().__init__()

        print("starting..")
        self.env = env
        self.test_env = test_env
        #print(env.observation_space.size())
        #print(env.action_space.size())

        self.agent = Normalagent(env, env.observation_space, env.action_space)
        #print("obs {} , action {} ".format(self.agent.obs_size, self.agent.action_size))

        # Set seed.
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)
        self.test_env.seed(2 ** 31 - 1 - seed)
        # torch.backends.cudnn.deterministic = True  # It harms a performance.
        # torch.backends.cudnn.benchmark = False  # It harms a performance.

        self.device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu")

        # Base Memory
        self.memory = LazyMultiStepMemory(capacity=memory_size, state_shape=self.env.observation_space.shape,
                                          device=self.device, gamma=gamma, multi_step=multi_step)

        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, 'model')
        self.summary_dir = os.path.join(log_dir, 'summary')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.train_return = RunningMeanStats(log_interval)

        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.best_eval_score = -np.inf
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.gamma_n = gamma ** multi_step
        self.start_steps = start_steps
        self.update_interval = update_interval
        self.target_update_interval = target_update_interval
        #self.use_per = use_per
        self.num_eval_steps = num_eval_steps
        self.max_episode_steps = max_episode_steps
        self.log_interval = log_interval
        self.eval_interval = eval_interval

        # Define networks.
        self.policy = CateoricalPolicy(
            self.agent.obs_size, self.agent.action_size).to(self.device)
        #print("policy ")

        self.online_critic = TwinnedQNetwork(
            self.agent.obs_size, self.agent.action_size).to(device=self.device)
        #print("critic ")

        self.target_critic = TwinnedQNetwork(
            self.agent.obs_size, self.agent.action_size).to(device=self.device).eval()
        #print("target ")

        # Copy parameters of the learning network to the target network.
        self.target_critic.load_state_dict(self.online_critic.state_dict())

        # Disable gradient calculations of the target network.
        disable_gradients(self.target_critic)

        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        self.q1_optim = Adam(self.online_critic.Q1.parameters(), lr=lr)
        self.q2_optim = Adam(self.online_critic.Q2.parameters(), lr=lr)

        # Target entropy is -log(1/|A|) * ratio (= maximum entropy * ratio).
        self.target_entropy = -np.log(1.0 / self.env.action_space.n) * target_entropy_ratio

        # We optimize log(alpha), instead of alpha.
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = Adam([self.log_alpha], lr=lr)


    def run(self):
        while True:
            self.train_episode()
            if self.steps > self.num_steps:
                break

    def is_update(self):
        #print("update")
        return self.steps % self.update_interval == 0 \
               and self.steps >= self.start_steps


    def explore(self, state):
        # Act with randomness.
        state = torch.ByteTensor(
            state[None, ...]).to(self.device).float() / 255.
        with torch.no_grad():
            action, _, _ = self.policy.sample(state)
        #print("act with random -model sample", action.item())
        return action.item()

    def exploit(self, state):
        # Act without randomness.
        state = torch.ByteTensor(
            state[None, ...]).to(self.device).float() / 255.
        with torch.no_grad():
            action = self.policy.act(state)
            #print("act without random -model act", action.item())
        return action.item()

    def update_target(self):
        self.target_critic.load_state_dict(self.online_critic.state_dict())
        #print("update_target")

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        curr_q1, curr_q2 = self.online_critic(states)
        curr_q1 = curr_q1.gather(1, actions.long())
        curr_q2 = curr_q2.gather(1, actions.long())
        return curr_q1, curr_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            _, action_probs, log_action_probs = self.policy.sample(next_states)
            next_q1, next_q2 = self.target_critic(next_states)
            next_q = (action_probs * (torch.min(next_q1, next_q2) - self.alpha * log_action_probs)).sum(dim=1, keepdim=True)

        assert rewards.shape == next_q.shape
        return rewards + (1.0 - dones) * self.gamma_n * next_q

    def calc_critic_loss(self, batch, weights):
        #print("calculate critic loss")
        curr_q1, curr_q2 = self.calc_current_q(*batch)
        target_q = self.calc_target_q(*batch)

        # TD errors for updating priority weights
        errors = torch.abs(curr_q1.detach() - target_q)

        # We log means of Q to monitor training.
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()

        # Critic loss is mean squared TD errors with priority weights.
        q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)

        return q1_loss, q2_loss, errors, mean_q1, mean_q2

    def calc_policy_loss(self, batch, weights):
        states, actions, rewards, next_states, dones = batch
        #print("calculate policy loss")

        # (Log of) probabilities to calculate expectations of Q and entropies.
        _, action_probs, log_action_probs = self.policy.sample(states)

        with torch.no_grad():
            # Q for every actions to calculate expectations of Q.
            q1, q2 = self.online_critic(states)
            q = torch.min(q1, q2)

        # Expectations of entropies.
        entropies = -torch.sum(action_probs * log_action_probs, dim=1, keepdim=True)

        # Expectations of Q.
        q = torch.sum(torch.min(q1, q2) * action_probs, dim=1, keepdim=True)

        # Policy objective is maximization of (Q + alpha * entropy) with
        # priority weights.
        policy_loss = (weights * (- q - self.alpha * entropies)).mean()

        return policy_loss, entropies.detach()

    def calc_entropy_loss(self, entropies, weights):
        assert not entropies.requires_grad
        #print("calculate policy loss")

        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropies)
            * weights)
        return entropy_loss

    def save_models(self, save_dir):
        super().save_models(save_dir)
        self.policy.save(os.path.join(save_dir, 'policy.pth'))
        self.online_critic.save(os.path.join(save_dir, 'online_critic.pth'))
        self.target_critic.save(os.path.join(save_dir, 'target_critic.pth'))

    def train_episode(self):
        self.episodes += 1
        episode_return = 0.
        episode_steps = 0

        done = False
        #state = self.env.reset()
        state = self.agent.convert_obs(self.env.reset())

        while (not done) and episode_steps <= self.max_episode_steps:

            if self.start_steps > self.steps:
                #action = self.agent.my_act(state)
                action = self.exploit(state)

            else:
                action = self.explore(state)

            #print("action {} ".format(action))
            next_state, reward, done, _ = self.env.step(self.agent.convert_act(action))


            next_state = self.agent.convert_obs(next_state)
            #reward = torch.tensor([reward], device=self.device, dtype=torch.float)
            #done = torch.tensor([done], device=self.device, dtype=torch.float)


            # Clip reward to [-1.0, 1.0].
            clipped_reward = max(min(reward, 1.0), -1.0)

            # To calculate efficiently, set priority=max_priority here.
            self.memory.append(state, action, clipped_reward, next_state, done)

            self.steps += 1
            episode_steps += 1
            episode_return += reward
            state = next_state

            if self.is_update():
                #print("go to learn()")
                self.learn()

            if self.steps % self.target_update_interval == 0:
                #print("steps % target_update_interval == 0  --> set 8000")
                self.update_target()

            if self.steps % self.eval_interval == 0:
                #print("steps % eval_interval == 0  --> set 1000")
                self.evaluate()
                self.save_models(os.path.join(self.model_dir, 'final'))

        # We log running mean of training rewards.
        self.train_return.append(episode_return)

        if self.episodes % self.log_interval == 0:
            self.writer.add_scalar(
                'reward/train', self.train_return.get(), self.steps)

        print(f'Episode: {self.episodes}  '
              f'Episode steps: {episode_steps}  '
              f'Return: {episode_return}')

    def learn(self):
        assert hasattr(self, 'q1_optim') and hasattr(self, 'q2_optim') and\
            hasattr(self, 'policy_optim') and hasattr(self, 'alpha_optim')

        self.learning_steps += 1
        #print("learn..")


        batch = self.memory.sample(self.batch_size)
        # Set priority weights to 1 when we don't use PER.
        weights = 1.

        q1_loss, q2_loss, errors, mean_q1, mean_q2 = self.calc_critic_loss(batch, weights)
        policy_loss, entropies = self.calc_policy_loss(batch, weights)
        entropy_loss = self.calc_entropy_loss(entropies, weights)

        update_params(self.q1_optim, q1_loss)
        update_params(self.q2_optim, q2_loss)
        update_params(self.policy_optim, policy_loss)
        update_params(self.alpha_optim, entropy_loss)

        self.alpha = self.log_alpha.exp()

        if self.learning_steps % self.log_interval == 0:
            self.writer.add_scalar(
                'loss/Q1', q1_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/Q2', q2_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/policy', policy_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/alpha', entropy_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'stats/alpha', self.alpha.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'stats/mean_Q1', mean_q1, self.learning_steps)
            self.writer.add_scalar(
                'stats/mean_Q2', mean_q2, self.learning_steps)
            self.writer.add_scalar(
                'stats/entropy', entropies.detach().mean().item(),
                self.learning_steps)

    def evaluate(self):

        #print("evaluate..")

        num_episodes = 0
        num_steps = 0
        total_return = 0.0

        while True:

            state = self.agent.convert_obs(self.test_env.reset())
            #state = self.test_env.reset()

            episode_steps = 0
            episode_return = 0.0
            done = False

            while (not done) and episode_steps <= self.max_episode_steps:

                action = self.exploit(state)


                next_state, reward, done, _ = self.test_env.step(self.agent.convert_act(action))
                next_state = self.agent.convert_obs(next_state)

                num_steps += 1
                episode_steps += 1
                episode_return += reward
                state = next_state

            num_episodes += 1
            total_return += episode_return

            if num_steps > self.num_eval_steps:
                break

        mean_return = total_return / num_episodes

        if mean_return > self.best_eval_score:
            self.best_eval_score = mean_return
            self.save_models(os.path.join(self.model_dir, 'best'))

        self.writer.add_scalar(
            'reward/test', mean_return, self.steps)
        print('-' * 60)
        print(f'Num steps: {self.steps}  '
              f'return: {mean_return:<5.3f}')
        print('-' * 60)


    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def __del__(self):
        self.env.close()
        self.test_env.close()
        self.writer.close()