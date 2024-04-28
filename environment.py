import gym


class Envionment:
    """
    A wrapper class around gym.env
    """

    def __init__(self) -> None:
        pass

    def reset(self):
        pass

    def step(self, action):
        pass


class CartPoleEnvironment(Envionment):
    def __init__(self, render_mode="") -> None:
        super().__init__()
        if render_mode:
            self.env = gym.make("CartPole-v1", render_mode=render_mode)
        else:
            self.env = gym.make("CartPole-v1")

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)


class PendulumEnvironment(Envionment):
    def __init__(self, render_mode="") -> None:
        super().__init__()
        if render_mode:
            self.env = gym.make("Pendulum-v1", render_mode=render_mode)
        else:
            self.env = gym.make("Pendulum-v1")

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)


class LunarLanderEnvironment(Envionment):
    def __init__(self, render_mode="") -> None:
        super().__init__()
        if render_mode:
            self.env = gym.make("LunarLander-v2", render_mode=render_mode)
        else:
            self.env = gym.make("LunarLander-v2")

    def reset(self):
        self.comp_steps = 0
        return self.env.reset()

    def step(self, action):
        state, reward, done, a, b = self.env.step(action)
        self.comp_steps += 1
        if done: reward = 200
        if self.comp_steps > 1000: 
            done = True
            print("---------------- ENV OVERSHOT -------------------")
        return (state, reward, done, a, b)
