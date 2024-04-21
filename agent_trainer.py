import torch.nn as nn
import torch

class AgentTrainer:
    def __init__(self, agent, env, optimizer, power_replay, logger, max_num_steps, epsilon, epsilon_decay, min_epsilon, gamma, sampler, device='cpu'):
        self.device = device
        self.agent = agent
        self.env = env
        self.optimizer = optimizer
        self.loss_fn = nn.MSELoss() # to be done MSE
        self.power_replay = power_replay
        self.logger = logger
        self.max_num_steps = num_steps
        self._steps_done = 1
        self._inference_mode = False
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.sampler = sampler


    def train_step(self):
        score = 0
        self.optimizer.zero_grad()
        state, _ = self.env.reset()
        
        for _ in range(self.num_steps):
            # action selection
            action = self.agent.selectAction(state, self.epsilon)
            next_state, reward, done, _ = self.env.step(action)

            # append to buffer
            self.power_replay.append((state, action, reward, next_state))
            state = next_state
            
            score += reward     # update score


            train_batch = self.sampler.sample_batch()
            state = self._get_states_tensor(train_batch, 0)
            next_state = self._get_states_tensor(train_batch, 3)

            # q estimates for training
            q_vals = self.agent.network(state.to(self.device)).detach()
            next_state_q_vals = self.agent.targetNetwork(next_state.to(self.device)).detach()

            # calc returns for training
            for i in range(len(train_batch)):
                q_estimates[i][train_batch[i][1]] = train_batch[i][2] + self.gamma * torch.max(next_state_q_vals[i]).item()


            train_ds = torch.utils.data.TensorDataset(state, q_estimates)
            train_dl = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

            self.agent.network.train()
            
            # train policy net
            tot_loss = 0.0
            for state, q_estimates in train_dl:
                out = self.agent.network(state.to(self.device))
                loss = self.loss_fn(out, q_estimates.to(self.device))
                tot_loss += loss.item()
                loss.backward()
                optimizer.step()

            self.agent.network.eval()   

            if done:
                break

            
        self.epsilon = self._decay_epsilon(self.epsilon, self.epsilon_decay, self.min_epsilon)


        return score

    def train_steps(self, num_steps):
        for _ in range(num_steps):
            score = self.train_step()
            self.logger.log("score", score, self._steps_done)
            self._steps_done += 1


            # update target netword ---> immedeately or after few steps??? 
            self.agent.targetNetwork.load_state_dict(self.agent.network)
            self.agent.targetNetwork.eval()



    def save(self, path):
        self.agent.targetNetwork.save(path)

    def load(self, path):
        self.agent.targetNetwork.load(path)

    def inference_mode(self, num_episodes):
        state, _ = self.env.reset()
        score = 0
        for _ in range(num_episodes):
            action = self.agent.selectAction(state)
            next_state, reward, done, _ = self.env.step(action)
            state = next_state
            score += reward
            if done:
                break

        return score

    def _get_states_tensor(sample, states_idx):
        sample_len = len(sample)
        states_tensor = torch.empty((sample_len, n_features), dtype=torch.float32, requires_grad=False)

        features_range = range(n_features)
        for i in range(sample_len):
            for j in features_range:
                states_tensor[i, j] = sample[i][states_idx][j].item()

        return states_tensor

    def _decay_epsilon(ep, ep_decay, min_ep):
        ep *= ep_decay
        return max(min_ep, ep)