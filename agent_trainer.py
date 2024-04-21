import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset

from transiton import Transition
from power_replay import PowerReplay


class AgentTrainer:
    """
    This class is used to train an agent in an environment. It uses a given agent, environment, logger, and optimizer.
    It also uses a power replay for the agent's memory and has parameters for the maximum number of steps, epsilon, epsilon decay,
    minimum epsilon, gamma, and device.
    """

    def __init__(
        self,
        agent,
        env,
        logger,
        optimizer,
        power_replay: PowerReplay,
        max_num_steps=1000,
        epsilon=1.0,
        epsilon_decay=0.99,
        min_epsilon=0.01,
        gamma=0.99,
        device="cpu",
        tau=0.1,
    ):

        self.device = torch.device(device)
        self.agent = agent
        self.env = env
        self.optimizer = optimizer
        self.loss_fn = nn.MSELoss()
        self.power_replay = power_replay
        self.logger = logger
        self.max_num_steps = max_num_steps
        self._steps_done = 1
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.episode_id = 0
        self.tau = tau

    def train_episode(self):
        """
        Train the agent for a single episode
        """

        score = 0
        self.optimizer.zero_grad()
        state, _ = self.env.reset()
        done = False
        total_next_qs = []

        while not done:
            # TODO handle addition for larger chunks

            action = self.agent.selectAction(state, self.epsilon)
            next_state, reward, done, _, _ = self.env.step(action)
            transition = Transition(state, action, reward, next_state)
            self.power_replay.addTransitions([transition], self.episode_id)
            state = next_state

            score += reward  # update score

            train_batch = self.power_replay.getBatch()
            states = self.get_states_tensor(train_batch)
            next_states = self.get_next_states_tensor(train_batch)

            # q estimates for training
            q_vals = self.agent.network(states.to(self.device)).detach()
            next_state_q_vals = self.agent.targetNetwork(
                next_states.to(self.device)
            ).detach()

            total_next_qs.append(next_state_q_vals.sum())

            # calc returns for training
            for i in range(len(train_batch)):
                q_vals[i][train_batch[i].action] = (
                    train_batch[i].reward
                    + self.gamma * torch.max(next_state_q_vals[i]).item()
                )

            train_ds = TensorDataset(states, q_vals)
            train_dl = DataLoader(
                train_ds, batch_size=self.power_replay.batch_size, shuffle=True
            )

            self.agent.network.train()

            # train policy net
            tot_loss = 0.0
            for st, q_estimates in train_dl:
                out = self.agent.network(st.to(self.device))
                loss = self.loss_fn(out, q_estimates.to(self.device))
                tot_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            self._steps_done += 1

            if done:
                break

        self.epsilon = self._decay_epsilon(
            self.epsilon, self.epsilon_decay, self.min_epsilon
        )

        return score, sum(total_next_qs) / len(total_next_qs)

    def train_steps(self, num_steps):
        """
        Train the agent for a number of steps
        """

        self.fill_buffer()

        while self._steps_done < num_steps:
            score, target_qs = self.train_episode()
            self.logger.log("score", score, self.episode_id)
            self.logger.log("target_qs", target_qs, self.episode_id)
            self.logger.log("steps", self._steps_done, self.episode_id)
            self.logger.log("epsilon", self.epsilon, self.episode_id)

            for target_param, policy_param in zip(
                self.agent.targetNetwork.parameters(), self.agent.network.parameters()
            ):
                target_param.data.copy_(
                    self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
                )

            self.agent.targetNetwork.eval()
            if self.episode_id % 20 == 0:
                score = self.inference_mode()
                self.logger.log("inference", score, self.episode_id)

            self.episode_id += 1

    def fill_buffer(self):
        while not self.power_replay.samplable():
            state, _ = self.env.reset()
            done = False
            while not done:
                action = self.agent.selectAction(state, self.epsilon)
                next_state, reward, done, _, _ = self.env.step(action)
                transition = Transition(state, action, reward, next_state)
                self.power_replay.addTransitions([transition], self.episode_id)
                state = next_state

    def save(self, path):
        """
        Save the target network to a file
        """
        self.agent.targetNetwork.save(path)

    def load(self, path):
        """
        Load the target network from a file
        """
        self.agent.targetNetwork.load(path)

    def inference_mode(self):
        """
        Run the agent in inference mode for a number of episodes
        """
        state, _ = self.env.reset()
        done = False
        score = 0
        with torch.no_grad():
            while not done:
                action = self.agent.selectAction(state)
                next_state, reward, done, _, _ = self.env.step(action)
                state = next_state
                score += reward

        return score

    def get_states_tensor(self, transitions):
        sample_len = len(transitions)  # number of samples in the batch
        n_features = len(transitions[0].state)  # number of states in the env

        states_tensor = torch.empty(
            (sample_len, n_features), dtype=torch.float32, requires_grad=False
        )

        for i in range(sample_len):
            for j in range(n_features):
                states_tensor[i, j] = float(transitions[i].state[j])

        return states_tensor

    def get_next_states_tensor(self, transitions):
        sample_len = len(transitions)  # number of samples in the batch
        n_features = len(transitions[0].state)  # number of states in the env

        states_tensor = torch.empty(
            (sample_len, n_features), dtype=torch.float32, requires_grad=False
        )

        for i in range(sample_len):
            for j in range(n_features):
                states_tensor[i, j] = float(transitions[i].nextState[j])

        return states_tensor

    def _decay_epsilon(self, ep, ep_decay, min_ep):
        ep *= ep_decay
        return max(min_ep, ep)
