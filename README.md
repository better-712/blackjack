#  Multi-Player Blackjack with Reinforcement Learning

## Milestone 1: State Space Specification

###  Motivation

This project aims to train a reinforcement learning agent to play Blackjack in a multi-player setting. Unlike traditional single-agent Blackjack environments, this version introduces multiple players sharing a finite deck of cards, which increases the uncertainty in future draws. This models real-world card games more accurately and introduces a more complex, partially observable environment for the agent.

---

###  Environment Setup

- **Game**: Blackjack (21)  
- **Number of players**: Configurable, currently set to 3 (including the agent)  
- **Decks**: Configurable, typically 6 decks (312 cards total)  
- **Rules**:
  - Each player is dealt two cards  
  - The dealer has one visible and one hidden card  
  - Players act sequentially  
  - Rewards are +1 for a win, 0 for a draw, -1 for a loss  

---

###  State Space Specification

Each player's state observation is structured as a dictionary:

```python
{
  "player_hand": [int],        # Sum of player's cards (or full card list)
  "dealer_visible": int,       # Dealer's visible card (1-10)
  "player_index": int          # The current player's index
}
```
### Action Space

Available actions (discrete):

- `0`: **Stand** – Stop drawing cards  
- `1`: **Hit** – Draw one more card  

---

###  Transition Function

- Player chooses **Hit** → draw a card → update hand value  
- If hand value > 21 → bust → move to next player  
- If **Stand** → directly move to next player  
- After all players act, dealer draws until reaching 17 or more  
- Game ends → reward is computed for each player  

---

###  Reward Function

Reward is computed at the end of each game **per player**:

| Outcome        | Reward |
|----------------|--------|
| Win            | `+1`   |
| Draw           | `0`    |
| Loss           | `-1`   |

## Algorithm: Deep Q-Network (DQN)

We used [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) to implement a Deep Q-Network (DQN) agent for our Blackjack environment.

### Training Configuration

```python
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500_000, callback=callback)
```
Environment: Single-player (against dealer, shared 6-deck shoe)
Implementation: Gymnasium-compatible environment

## Evaluation Results

After training, the model was saved to disk as `dqn_blackjack.zip` and evaluated over **1,000 episodes**.

### Performance Summary

| Outcome | Count | Rate     |
|---------|-------|----------|
| Wins    | 435   | 43.50%   |
| Draws   | 101    | 1.01%    |
| Losses  | 464   | 4640%   |

| Strategy         | Wins | Draws | Losses | Win Rate |
|------------------|------|-------|--------|----------|
| **DQN Agent**     | 435  | 101    | 464    | 43.50%   |
| **Baseline** (`Hit < 17`) | 412  | 116   | 472    | 41.20%   |
| **Random Policy** | 276  | 41    | 683    | 27.60%   |
