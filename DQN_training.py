import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
from blackjack import BlackjackEnv


class TQDMCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress")

    def _on_step(self):
        self.pbar.update(self.model.n_envs)
        return True

    def _on_training_end(self):
        self.pbar.close()


def make_env():
    env = BlackjackEnv(num_players=1, num_decks=6)
    env = FlattenObservation(env)
    return env


def train_agent():
    env = make_env()

    from stable_baselines3 import DQN

    model = DQN("MlpPolicy", env, verbose=1)
    callback = TQDMCallback(total_timesteps=500_000)
    model.learn(total_timesteps=500_000, callback=callback)
    model.save("dqn_blackjack")
    print("Model saved as dqn_blackjack")

    return model


def evaluate(model, episodes=1000):
    env = make_env()
    wins = 0
    draws = 0
    losses = 0

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        if reward == 1:
            wins += 1
        elif reward == 0:
            draws += 1
        else:
            losses += 1

    print(f"\nEvaluation results over {episodes} episodes:")
    print(f"Wins: {wins} ({wins / episodes * 100:.2f}%)")
    print(f"Draws: {draws} ({draws / episodes * 100:.2f}%)")
    print(f"Losses: {losses} ({losses / episodes * 100:.2f}%)")


if __name__ == "__main__":
    model = train_agent()
    evaluate(model, episodes=1000)
