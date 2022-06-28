import pybullet_envs
import gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F
from torch.optim import Adam
import random
ENV_NAME = "Walker2DBulletEnv-v0"

LAMBDA = 0.97
GAMMA = 0.99
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACTOR_LR = 3e-4
CRITIC_LR = 2e-4


CLIP = 0.2
ENTROPY_COEF = 1e-2
BATCHES_PER_UPDATE = 64
BATCH_SIZE = 64

MIN_TRANSITIONS_PER_UPDATE = 2048
MIN_EPISODES_PER_UPDATE = 4

ITERATIONS = 5000


# https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
# https://arxiv.org/pdf/1707.06347.pdf L_CLIP LOSS
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Advice: use same log_sigma for all states to improve stability
        # You can do this by defining log_sigma as nn.Parameter(torch.zeros(...))
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, action_dim))

        self.sigma = nn.Parameter(torch.zeros(1, action_dim))

    def compute_proba(self, state, action):
        # Returns probability of action according to current policy and distribution of actions
        actor_means = self.model(state)  # [batch_size x action_dim]
        sigma = torch.exp(self.sigma).expand_as(actor_means)  # [batch_size x action_dim]
        d = Normal(actor_means, sigma)
        return torch.exp(d.log_prob(action)).sum(-1), d  # [batch_size], ~Normal(mu, sigma)

    def act(self, state: torch.tensor):
        # Returns an action (with tanh), not-transformed action (without tanh) and distribution of non-transformed actions
        # Remember: agent is not deterministic, sample actions from distribution (e.g. Gaussian)

        actor_means = self.model(state)  # [batch_size x action_dim]
        sigma = torch.exp(self.sigma).expand_as(actor_means)  # [batch_size x action_dim]
        d = Normal(actor_means, sigma)
        action = d.sample()  # [batch_size x action_dim]
        return torch.tanh(action), action, d

    def loss(self, state, action, old_prob, advantage, eps):
        new_prob, d = self.compute_proba(state, action)
        entropy = d.entropy().mean()

        clip_advantage = torch.clip(advantage, 1 - eps, 1 + eps) * advantage
        loss = -torch.min((new_prob / old_prob) * advantage, clip_advantage).mean() - ENTROPY_COEF * entropy

        return loss


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )

    def get_value(self, state):
        return self.model(state)

    def loss(self, state, target_value):
        pred_value = self.get_value(state)

        return F.mse_loss(pred_value.view(-1), target_value)


def evaluate_policy(env, agent, episodes=5):
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.

        while not done:
            state, reward, done, _ = env.step(agent.act(state)[0])
            total_reward += reward
        returns.append(total_reward)
    return returns


def compute_lambda_returns_and_gae(trajectory):
    lambda_returns = []
    gae = []
    last_lr = 0.
    last_v = 0.
    for _, _, r, _, v in reversed(trajectory):
        ret = r + GAMMA * (last_v * (1 - LAMBDA) + last_lr * LAMBDA)
        last_lr = ret
        last_v = v
        lambda_returns.append(last_lr)  # r
        gae.append(last_lr - v)  # r + gamma * V(next_state) - V_state

    # Each transition contains state, action, old action probability, value estimation and advantage estimation
    return [(s, a, p, v, adv) for (s, a, _, p, _), v, adv in zip(trajectory, reversed(lambda_returns), reversed(gae))]


def sample_episode(env, agent):
    s = env.reset()
    d = False
    trajectory = []
    while not d:
        a, pa, p = agent.act(s)
        v = agent.get_value(s)
        ns, r, d, _ = env.step(a)
        trajectory.append((s, pa, r, p, v))
        s = ns
    return trajectory

    # return compute_lambda_returns_and_gae(trajectory)

    # state = np.array(state)
    # action = np.array(action)
    # old_prob = np.array(old_prob)
    # target_value = np.array(target_value)
    # advantage = np.array(advantage)


class PPO:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(DEVICE)
        self.critic = Critic(state_dim).to(DEVICE)

        self.actor_optim = Adam(self.actor.parameters(), ACTOR_LR)
        self.critic_optim = Adam(self.critic.parameters(), CRITIC_LR)

    def update(self, trajectories):
        transitions = [t for traj in trajectories for t in traj]  # Turn a list of trajectories into list of transitions
        state, action, old_prob, target_value, advantage = map(np.array, zip(*transitions))

        advnatage = (advantage - advantage.mean()) / (advantage.std() + 1e-16)

        for _ in range(BATCHES_PER_UPDATE):
            idx = np.random.randint(0, len(transitions), BATCH_SIZE)  # Choose random batch

            s = torch.tensor(state[idx], device=DEVICE).float()
            a = torch.tensor(action[idx], device=DEVICE).float()
            op = torch.tensor(old_prob[idx],
                              device=DEVICE).float()  # Probability of the action in state s.t. old policy
            v = torch.tensor(target_value[idx], device=DEVICE).float()  # Estimated by lambda-returns
            adv = torch.tensor(advantage[idx], device=DEVICE).float()  # Estimated by generalized advantage estimation

            # TODO: Update actor here
            actor_loss = self.actor.loss(s, a, op, adv, CLIP)

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            # TODO: Update critic here
            critic_loss = self.critic.loss(s, v)

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

    def get_value(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state]), device=DEVICE).float()
            value = self.critic.get_value(state)
        return value.cpu().item()

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state]), device=DEVICE).float()
            action, pure_action, distr = self.actor.act(state)
            prob = torch.exp(distr.log_prob(pure_action).sum(-1))
        return action.cpu().numpy()[0], pure_action.cpu().numpy()[0], prob.cpu().item()

    def save(self, name="agent"):
        torch.save(self.actor, f"{name}.pkl")


if __name__ == "__main__":
    env = gym.make(ENV_NAME)
    seed = 0

    env.seed(seed)
    env.action_space.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ppo = PPO(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
    state = env.reset()
    episodes_sampled = 0
    steps_sampled = 0
    best_reward = -np.inf

    for i in range(ITERATIONS):
        trajectories = []
        steps_ctn = 0

        while len(trajectories) < MIN_EPISODES_PER_UPDATE or steps_ctn < MIN_TRANSITIONS_PER_UPDATE:
            traj = sample_episode(env, ppo)
            steps_ctn += len(traj)
            trajectories.append(traj)
        episodes_sampled += len(trajectories)
        steps_sampled += steps_ctn

        ppo.update(trajectories)

        if (i + 1) % (ITERATIONS // 100) == 0:
            rewards = evaluate_policy(env, ppo, 5)
            print(
                f"Step: {i + 1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}, Episodes: {episodes_sampled}, Steps: {steps_sampled}")
            if best_reward < np.mean(rewards):
                best_reward = np.mean(rewards)
                ppo.save()
