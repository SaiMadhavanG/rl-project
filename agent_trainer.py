import torch.nn as nn
import torch

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
        power_replay, 
        max_num_steps   =   1000, 
        epsilon         =   1.0, 
        epsilon_decay   =   0.99, 
        min_epsilon     =   0.01, 
        gamma           =   0.99,
        device          =   'cpu'
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


    def train_step(self):
        """
        Train the agent for a single step
        """

        score = 0
        self.optimizer.zero_grad()
        state, _ = self.env.reset()
        done = False

        while done:
            # action selection
            action = self.agent.selectAction(state, self.epsilon)
            next_state, reward, done, _ = self.env.step(action)

            # append to buffer
            self.power_replay.append((state, action, reward, next_state))
            state = next_state
            
            score += reward     # update score

            train_batch = self.power_replay.getBatch()
            state = self._get_states_tensor(train_batch, 0)
            next_state = self._get_states_tensor(train_batch, 3)

            # q estimates for training
            q_vals = self.agent.network(state.to(self.device)).detach()
            next_state_q_vals = self.agent.targetNetwork(next_state.to(self.device)).detach()

            # calc returns for training
            for i in range(len(train_batch)):
                q_vals[i][train_batch[i][1]] = train_batch[i][2] + self.gamma * torch.max(next_state_q_vals[i]).item()


            train_ds = torch.utils.data.TensorDataset(state, q_vals)
            train_dl = torch.utils.data.DataLoader(train_ds, batch_size=self.power_replay.batch_size, shuffle=True)

            self.agent.network.train()
            
            # train policy net
            tot_loss = 0.0
            for state, q_estimates in train_dl:
                out = self.agent.network(state.to(self.device))
                loss = self.loss_fn(out, q_estimates.to(self.device))
                tot_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            self.agent.network.eval()
            
            if done:
                break
        
        self.epsilon = self._decay_epsilon(self.epsilon, self.epsilon_decay, self.min_epsilon)


        return score

    def train_steps(self, num_steps):
        """
        Train the agent for a number of steps
        """
        
        tau = 0.05      # target network update rate

        for _ in range(num_steps):
            score = self.train_step()
            self.logger.log("score", score, self._steps_done)
            self._steps_done += 1

            for target_param, policy_param in zip(self.agent.targetNetwork.parameters(), self.agent.network.parameters()):
                target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)
            
            self.agent.targetNetwork.eval()



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

    def inference_mode(self, num_episodes):
        """
        Run the agent in inference mode for a number of episodes
        """
        state, _ = self.env.reset()
        score = 0
        with torch.no_grad():
            for _ in range(num_episodes):
                action = self.agent.selectAction(state)
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                score += reward
                if done:
                    break

        return score

    def _get_states_tensor(self, sample, states_idx):
        sample_len = len(sample)        # number of samples in the batch
        n_features = len(sample[0])     # number of states in the env

        states_tensor = torch.empty((sample_len, n_features), dtype=torch.float32, requires_grad=False)

        for i in range(sample_len):
            for j in range(n_features):
                states_tensor[i, j] = sample[i][states_idx][j].item()
        return states_tensor

    def _decay_epsilon(self, ep, ep_decay, min_ep):
        ep *= ep_decay
        return max(min_ep, ep)