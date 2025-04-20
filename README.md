# Multi-Player Blackjack with Reinforcement Learning

## Milestone 1: State Space Specification

### 🎯 Motivation

This project aims to train a reinforcement learning agent to play Blackjack in a multi-player setting. Unlike traditional single-agent Blackjack environments, this version introduces multiple players sharing a finite deck of cards, which increases the uncertainty in future draws. This models real-world card games more accurately and introduces a more complex, partially observable environment for the agent.

### 🕹️ Environment Setup

- **Game**: Blackjack (21)  
- **Number of players**: Configurable, currently set to 3 (including the agent)  
- **Decks**: Configurable, typically 6 decks (312 cards total)  
- **Rules**:
  - Each player is dealt two cards  
  - The dealer has one visible and one hidden card  
  - Players act sequentially  
  - Rewards are +1 for a win, 0 for a draw, -1 for a loss  

---

### 🧠 State Space Specification

Each player's state observation is structured as a dictionary:

```python
{
  "player_hand": [int],        # Sum of player's cards (or full card list)
  "dealer_visible": int,       # Dealer's visible card (1-10)
  "player_index": int          # The current player's index
}
