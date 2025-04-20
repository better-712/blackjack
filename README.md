Multi-Player Blackjack with Reinforcement Learning

Milestone 1: State Space Specification

Motivation

This project aims to train a reinforcement learning agent to play Blackjack in a multi-player setting. Unlike traditional single-agent Blackjack environments, this version introduces multiple players sharing a finite deck of cards, which increases the uncertainty in future draws. This models real-world card games more accurately and introduces a more complex, partially observable environment for the agent.

Environment Setup

Game: Blackjack (21)

Number of players: Configurable, currently set to 3 (including the agent)

Decks: Configurable, typically 6 decks (312 cards total)

Rules: Standard Blackjack rules

Each player is dealt two cards

The dealer has one visible and one hidden card

Players act sequentially

Rewards are +1 for a win, 0 for a draw, -1 for a loss

State Space Specification

The state representation for each player consists of the following components:

Observation (Dict format)

player_hand (Box): The sum of the player’s current hand (optionally full card list)

dealer_visible (Discrete): The dealer's visible card (1-10)

player_index (Discrete): Index of the current player (used in shared policy setups)

Optional future extensions:

has_usable_ace (Bool): Whether the player has a usable ace

cards_remaining (Box or vector): A histogram representing remaining cards in the deck

State Characteristics

Partial observability: Players do not observe each other's hands or actions

Stochastic transitions: Due to card draws

Sequential actions: Players act in turn

Action Space

Hit: Draw another card

Stand: Stop drawing cards

(Future work may extend to include Double Down or Split actions)

Transition Function

Depends on current hand value and the drawn card

The dealer draws after all players have acted

Game ends when all players and dealer have completed their turns

Reward Function

Computed at the end of the round per player:

Win: +1

Draw: 0

Loss or bust: -1

Implementation Plan

The Blackjack environment is implemented in Python, following the Gymnasium interface to support SB3 training. The state space and transition logic are integrated with card draw mechanics and multiplayer sequencing.

Repository

All code is maintained at: https://github.com/better-712/blackjack

Milestone Summary

This README outlines the completed "State Space Specification" milestone. The next milestone will focus on full implementation and integration with RL agents (e.g., DQN, PPO) and reward tracking.

