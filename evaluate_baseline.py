from blackjack import BlackjackEnv
if __name__ == '__main__':
    env = BlackjackEnv(num_players=1, render_mode=None)
    def baseline_policy(obs):
        player_sum = obs["player_hand"][0]
        if player_sum < 17:
            return 1  # Hit
        else:
            return 0  # Stand
    num_episodes = 1000
    win, draw, loss = 0, 0, 0

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            action = baseline_policy(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        if reward == 1:
            win += 1
        elif reward == 0:
            draw += 1
        else:
            loss += 1

    print(f"Baseline strategy in {num_episodes} games：")
    print(f"Win：{win} games ({win / num_episodes:.2%})")
    print(f"Draw：{draw} games ({draw / num_episodes:.2%})")
    print(f"Loss：{loss} games ({loss / num_episodes:.2%})")
