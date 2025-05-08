import numpy as np
from blackjack import BlackjackEnv
from gymnasium.wrappers import FlattenObservation
import random

def make_env(num_players=1):
    env = BlackjackEnv(num_players=num_players, num_decks=6)
    return FlattenObservation(env)

def evaluate_random_policy(episodes=1000):
    env = make_env(num_players=1)
    wins, draws, losses = 0, 0, 0

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            action = random.choice([0, 1])  # 0 = Stand, 1 = Hit
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        if reward == 1:
            wins += 1
        elif reward == 0:
            draws += 1
        else:
            losses += 1

    print(f"\nRandom policy evaluation over {episodes} episodes:")
    print(f"Wins:   {wins} ({wins / episodes * 100:.2f}%)")
    print(f"Draws:  {draws} ({draws / episodes * 100:.2f}%)")
    print(f"Losses: {losses} ({losses / episodes * 100:.2f}%)")

if __name__ == "__main__":
    evaluate_random_policy(episodes=1000)
