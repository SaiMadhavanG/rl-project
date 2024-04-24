import torch.nn as nn
import torch
import torch.nn.functional as F
import os

from transiton import Transition
from power_replay import PowerReplay
from tracker import Tracker


def createDir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)


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
        update_frequency=3,
        render_env=None,
    ):

        self.device = torch.device(device)
        self.agent = agent
        self.env = env
        self.optimizer = optimizer
        self.power_replay = power_replay
        self.logger = logger
        self.max_num_steps = max_num_steps
        self._steps_done = 1
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.episode_id = 0
        self.update_frequency = update_frequency
        self.render_env = render_env if render_env else self.env
        self.checkpointDir = "checkpoints/" + self.logger.exptId
        self.tracker = Tracker(self.power_replay.buffer)

    def my_loss(self, y_pred, y_true):
        # Calculate mean squared error loss only for the actions taken
        s1, s2 = y_true.shape
        loss = F.mse_loss(
            y_pred[torch.arange(s1), self.actionsHistory],
            y_true[torch.arange(s1), self.actionsHistory],
        )
        return loss

    def calculate_tde(self, transition):
        with torch.no_grad():
            Q_S_ = self.agent.targetNetwork(
                torch.tensor(transition.nextState).to(self.device)
            )
            if transition.terminated:
                y = transition.reward
            else:
                # Calculate target Q-value using target network
                y = transition.reward + self.gamma * torch.max(Q_S_)
            QS = self.agent.network(torch.tensor(transition.state).to(self.device))
            return (y - QS[transition.action].item()) ** 2

    def train_episode(self):
        """
        Train the agent for a single episode
        """

        score = 0
        state, _ = self.env.reset()
        done = False
        total_next_qs = []

        while not done:
            # TODO handle addition for larger chunks

            action = self.agent.selectAction(
                state, epsilon=self.epsilon, network="current"
            )
            next_state, reward, done, _, _ = self.env.step(action)
            transition = Transition(state, action, reward, next_state, done)
            tde_ = self.calculate_tde(transition)
            transition.setTDE(tde_)
            self.power_replay.addTransitions([transition], self.episode_id)
            self.tracker.set_tde(self.power_replay.buffer.chunks[-1])
            state = next_state

            score += reward  # update score

            self.power_replay.sweep()

            train_batch, train_chunks = self.power_replay.getBatch()
            currentStateBatch = self.get_states_tensor(train_batch)
            nextStateBatch = self.get_next_states_tensor(train_batch)

            # q estimates for training
            self.agent.targetNetwork.eval()
            Q_S_ = self.agent.targetNetwork(
                nextStateBatch.to(self.device)
            )  # Compute Q-values for next states

            Y = torch.zeros((self.power_replay.batch_size, self.agent.numActions)).to(
                self.device
            )
            QS = self.agent.network(currentStateBatch.to(self.device))
            self.actionsHistory = []
            rewards_batch = []
            for idx, transition in enumerate(train_batch):
                rewards_batch.append(reward)
                if transition.terminated:
                    y = transition.reward
                else:
                    # Calculate target Q-value using target network
                    y = transition.reward + self.gamma * torch.max(Q_S_[idx])
                self.actionsHistory.append(transition.action)
                Y[idx, transition.action] = y
                tde = (y - QS[idx, transition.action].item()) ** 2
                transition.setTDE(tde)

            for chunk in train_chunks:
                self.tracker.set_tde(chunk)

            # Set network back to training mode
            self.agent.network.train()

            # Zero out gradients
            self.optimizer.zero_grad()

            # Compute Q-values for current states

            # Compute loss and backpropagate
            loss = self.my_loss(QS, Y)
            loss.backward()
            self.optimizer.step()  # Update network parameters

            total_next_qs.append(Q_S_.flatten().sum().item())
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

            # for target_param, policy_param in zip(
            #     self.agent.targetNetwork.parameters(), self.agent.network.parameters()
            # ):
            #     target_param.data.copy_(
            #         self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            #     )

            if self.episode_id % self.update_frequency == 0:
                self.agent.targetNetwork.load_state_dict(
                    self.agent.network.state_dict()
                )

            self.agent.targetNetwork.eval()
            if self.episode_id % 5 == 0:
                score = self.inference_mode()
                self.logger.log("inference", score, self.episode_id)

            if self.episode_id % 50 == 0:
                self.save()

            self.episode_id += 1

    def fill_buffer(self):
        while not self.power_replay.samplable():
            state, _ = self.env.reset()
            done = False
            while not done:
                action = self.agent.selectAction(state, epsilon=1)
                next_state, reward, done, _, _ = self.env.step(action)
                transition = Transition(state, action, reward, next_state, done)
                self.power_replay.addTransitions([transition], self.episode_id)
                state = next_state

    def save(self):
        """
        Save the target network to a file
        """
        createDir(self.checkpointDir)
        checkpointPath = os.path.join(
            self.checkpointDir, f"checkpoint-{self.episode_id}.pth"
        )
        torch.save(self.agent.targetNetwork.state_dict(), checkpointPath)

    def load(self, path):
        """
        Load the target network from a file
        """
        self.agent.targetNetwork.load(path)

    def inference_mode(self):
        """
        Run the agent in inference mode for a number of episodes
        """
        state, _ = self.render_env.reset()
        done = False
        score = 0
        with torch.no_grad():
            while not done:
                action = self.agent.selectAction(state)
                next_state, reward, done, _, _ = self.render_env.step(action)
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
        ep -= ep_decay
        return max(min_ep, ep)
