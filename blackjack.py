import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import deque

class BlackjackEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, num_players=1, num_decks=6, render_mode=None):
        super().__init__()
        self.num_players = num_players
        self.num_decks = num_decks
        self.render_mode = render_mode

        self.deck = deque()
        self.players = [[] for _ in range(num_players)]
        self.dealer = []
        self.current_player = 0
        self.done = False
        self.rewards = [0 for _ in range(num_players)]

        self.observation_space = spaces.Dict({
            "player_hand": spaces.Box(low=0, high=31, shape=(1,), dtype=np.int32),
            "dealer_visible": spaces.Discrete(11),  # 1–10
            "player_index": spaces.Discrete(num_players)
        })

        self.action_space = spaces.Discrete(2)  # 0 = stand, 1 = hit
        self._init_deck()  

    def _init_deck(self):
        single_deck = [1,2,3,4,5,6,7,8,9,10,10,10,10] * 4
        self.deck = deque(single_deck * self.num_decks)
        random.shuffle(self.deck)

    def _draw_card(self):
        if len(self.deck) < 15:  
            self._init_deck()
        return self.deck.popleft()

    def _hand_value(self, hand):
        value = sum(hand)
        aces = hand.count(1)
        while value + 10 <= 21 and aces:
            value += 10
            aces -= 1
        return value

    def _is_bust(self, hand):
        return self._hand_value(hand) > 21

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.players = [[] for _ in range(self.num_players)]
        self.dealer = []
        self.rewards = [0 for _ in range(self.num_players)]
        self.done = False
        self.current_player = 0

        for _ in range(2):
            for p in self.players:
                p.append(self._draw_card())
            self.dealer.append(self._draw_card())

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), {}

    def _get_obs(self):
        hand_value = self._hand_value(self.players[self.current_player])
        dealer_card = self.dealer[0]
        return {
            "player_hand": np.array([hand_value], dtype=np.int32),
            "dealer_visible": dealer_card,
            "player_index": self.current_player
        }

    def step(self, action):
        if self.done:
            raise Exception("Game is over. Please reset.")

        player_hand = self.players[self.current_player]
        if action == 1:  # hit
            player_hand.append(self._draw_card())
            if self._is_bust(player_hand):
                self.rewards[self.current_player] = -1
                self._advance_player()
        else:  # stand
            self._advance_player()

        if self.done:
            self._play_dealer()
            self._compute_rewards()
            reward = self.rewards[0]  
            if self.render_mode == "human":
                self.render()
            return self._dummy_obs(), reward, True, False, {}

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), 0.0, False, False, {}

    def _advance_player(self):
        self.current_player += 1
        if self.current_player >= self.num_players:
            self.done = True

    def _play_dealer(self):
        while self._hand_value(self.dealer) < 17:
            self.dealer.append(self._draw_card())

    def _compute_rewards(self):
        dealer_value = self._hand_value(self.dealer)
        dealer_bust = self._is_bust(self.dealer)

        for i, hand in enumerate(self.players):
            if self.rewards[i] == -1:
                continue  # already bust
            player_value = self._hand_value(hand)
            if dealer_bust or player_value > dealer_value:
                self.rewards[i] = 1
            elif player_value == dealer_value:
                self.rewards[i] = 0
            else:
                self.rewards[i] = -1

    def _dummy_obs(self):
        return {
            "player_hand": np.array([0], dtype=np.int32),
            "dealer_visible": 0,
            "player_index": 0
        }

    def render(self):
        print(f"Dealer: {self.dealer} ({self._hand_value(self.dealer)})")
        for i, hand in enumerate(self.players):
            print(f"Player {i}: {hand} ({self._hand_value(hand)})")

    def close(self):
        pass
