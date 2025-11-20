import random
import numpy as np
from gymnasium import Env, spaces
from treys import Card, Deck, Evaluator
from PokerEquity import estimate_equity

# ======================
# Políticas de oponentes
# ======================

def cards_int_to_str(cards_int):
    """
    Convierte lista de ints de treys a lista de strings tipo 'As','Kd'
    """
    return [Card.int_to_str(c) for c in cards_int]

def policy_player1(hero_hand, board, num_opponents):
    hero_str = cards_int_to_str(hero_hand)
    eq = estimate_equity(hero_str, board_str=[], num_opponents=num_opponents, iters=2000)['win_prob']
    return 1 if eq > 0.20 else 0

def policy_player2(hero_hand, board, num_opponents):
    hero_str = cards_int_to_str(hero_hand)
    eq = estimate_equity(hero_str, board_str=[], num_opponents=num_opponents, iters=2000)['win_prob']
    return random.choice([1,2]) if eq > 0.30 else 0

def policy_player3(hero_hand, board, num_opponents):
    hero_str = cards_int_to_str(hero_hand)
    eq = estimate_equity(hero_str, board_str=[], num_opponents=num_opponents, iters=2000)['win_prob']
    return random.choice([0,1]) if eq > 0.20 else 0

def policy_player4(hero_hand, board, num_opponents):
    hero_str = cards_int_to_str(hero_hand)
    eq = estimate_equity(hero_str, board_str=[], num_opponents=num_opponents, iters=2000)['win_prob']
    if eq > 0.15:
        return 1
    else:
        return 0 if random.random() < 0.7 else 1

# ======================
# Entorno Gymnasium
# ======================

class Poker5EnvFull(Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    ACTIONS = ["fold", "call", "bet", "raise"]

    def __init__(self, opponent_policies=None, starting_stack=1000, small_blind=10, big_blind=20):
        super().__init__()

        self.dealer_pos = 0  # posición del botón, rota cada mano
        self.num_players = 5
        self.agent_id = 0
        self.agent_folded = False
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind

        # Políticas oponentes
        if opponent_policies is None:
            self.opponent_policies = [
                lambda obs: policy_player1(obs["hero_hand"], obs["board"], num_opponents=4),
                lambda obs: policy_player2(obs["hero_hand"], obs["board"], num_opponents=4),
                lambda obs: policy_player3(obs["hero_hand"], obs["board"], num_opponents=4),
                lambda obs: policy_player4(obs["hero_hand"], obs["board"], num_opponents=4),
            ]
        else:
            assert len(opponent_policies) == self.num_players - 1
            self.opponent_policies = opponent_policies

        self.action_space = spaces.Discrete(len(self.ACTIONS))

        # Observación simplificada
        self.observation_space = spaces.Dict({
            "hero_hand": spaces.MultiDiscrete([52, 52]),
            "board": spaces.MultiDiscrete([52]*5),
            "stacks": spaces.Box(low=0, high=np.inf, shape=(self.num_players,), dtype=np.float32),
            "pot": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "current_bet": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        })

        self.reset()

    # ======================
    # Funciones internas
    # ======================

    def reset(self, seed=None, options=None):
        self.deck = Deck()
        self.deck.shuffle()
        self.evaluator = Evaluator()
        self.board = []
        self.stacks = [self.starting_stack for _ in range(self.num_players)]
        self.pot = 0
        self.current_bet = self.big_blind
        self.bets = [0]*self.num_players
        self.hands = [None] * self.num_players
        self.active_players = [True]*self.num_players
        self.agent_folded = False
        self.dealer_pos = 0

        return self._get_obs(), {}

    def partial_reset(self):
        """Reset parcial de la mano sin tocar stacks ni baraja."""
        self.deck.shuffle()
        
        self.hands = [self.deck.draw(2) for _ in range(self.num_players)]

        self.active_players = [True] * self.num_players

        self.bets = [0] * self.num_players
        self.current_bet = self.big_blind
        self.pot = 0

        self.round_stage = 'preflop'
        self.agent_folded = False
        
        # === Blinds rotativos ===
        sb_pos = (self.dealer_pos + 1) % self.num_players
        bb_pos = (self.dealer_pos + 2) % self.num_players

        self.stacks[sb_pos] -= self.small_blind
        self.stacks[bb_pos] -= self.big_blind
        self.bets[sb_pos] = self.small_blind
        self.bets[bb_pos] = self.big_blind
        self.pot = self.small_blind + self.big_blind

        # Rotar dealer para la próxima mano
        self.dealer_pos = (self.dealer_pos + 1) % self.num_players

        self.reward = 0
        self.done = False

        # Retornar la observación inicial de la mano
        return self._get_obs()

    def _get_obs(self):
        board = self.board + [0]*(5-len(self.board))
        return {
            "hero_hand": self.hands[self.agent_id],
            "board": board,
            "stacks": np.array(self.stacks, dtype=np.float32),
            "pot": np.array([self.pot], dtype=np.float32),
            "current_bet": np.array([self.current_bet], dtype=np.float32)
        }

    def step(self, action):     
        done = False
        reward = 0

        # Acción agente
        self._apply_action(self.agent_id, action)
        
        if sum(self.active_players) == 1:
            done = True

        # Acción oponentes
        for i, policy in enumerate(self.opponent_policies):
            player_id = i+1
            if self.active_players[player_id] and not done:
                opp_action = policy(self._get_obs())
                self._apply_action(player_id, opp_action)
                if sum(self.active_players) == 1:
                    done = True
                    break

        # Avanzar ronda
        if not done:
            self.current_bet = 0
            if self.round_stage == 'preflop':
                self.board += self.deck.draw(3)  # flop
                self.round_stage = 'flop'
            elif self.round_stage == 'flop':
                self.board += self.deck.draw(1)  # turn
                self.round_stage = 'turn'
            elif self.round_stage == 'turn':
                self.board += self.deck.draw(1)  # river
                self.round_stage = 'river'

            done = self.round_stage == 'river'
        
        if done:
            reward = self._resolve_hand()

        return self._get_obs(), reward, done, False, {}

    def _apply_action(self, player_id, action):
        if not self.active_players[player_id]:
            return

        # Acción fold
        if action == 0:
            self.active_players[player_id] = False
            if player_id == self.agent_id:
                self.agent_folded = True
        # Acción call
        elif action == 1:
            to_call = self.current_bet - self.bets[player_id]
            to_call = min(to_call, self.stacks[player_id])
            self.stacks[player_id] -= to_call
            self.bets[player_id] += to_call
            self.pot += to_call
        # Acción bet
        elif action == 2:
            if self.current_bet == 0:
                to_bet = self.big_blind
            else:
                to_bet = self.current_bet
            to_bet = min(to_bet, self.stacks[player_id])
            self.stacks[player_id] -= to_bet
            self.bets[player_id] += to_bet
            self.pot += to_bet
            self.current_bet = self.bets[player_id]
        # Acción raise
        elif action == 3:
            if self.current_bet == 0:
                to_raise = 2*self.big_blind
            else:
                to_raise = 2*self.current_bet
            to_raise = min(to_raise, self.stacks[player_id])
            self.stacks[player_id] -= to_raise
            self.bets[player_id] += to_raise
            self.pot += to_raise
            self.current_bet = self.bets[player_id]

    def _resolve_hand(self):
        active_hands = [(i, self.hands[i]) for i, active in enumerate(self.active_players) if active]

        # Si solo queda un jugador, gana automáticamente
        if len(active_hands) == 1:
            winner_id, winner_hand = active_hands[0]
            self.winners = [winner_id]
            self.stacks[winner_id] += self.pot  # adjudicar pot

        else:
            scores = [(i, self.evaluator.evaluate(hand, self.board)) for i, hand in active_hands]
            min_score = min([s for i,s in scores])
            winners = [i for i,s in scores if s==min_score]
            self.winners = winners

            share = self.pot // len(winners)   # dividir el pot entre ganadores
            for w in winners:
                self.stacks[w] += share        # sumar fichas al stack del ganador
            self.pot = 0                       # limpiar el pot

        stack_change = self.stacks[self.agent_id] - self.starting_stack

        # Equity final
        hero_str = cards_int_to_str(self.hands[self.agent_id])
        board_str = cards_int_to_str(self.board)
        agent_equity_final = estimate_equity(hero_str, board_str=board_str, num_opponents=len(active_hands)-1, iters=2000)['win_prob']
        
        reward = stack_change / self.big_blind
        
        # Penaliza menos si perdió con buena mano
        if self.agent_id not in self.winners and stack_change < 0:
            reward -= (1 - agent_equity_final)

        # Penaliza o recompensa fold
        if self.agent_folded and not len(active_hands):
            hands_final = active_hands
            hands_final.append((self.agent_id, self.hands[self.agent_id]))
            scores_final = [(i, self.evaluator.evaluate(hand, self.board)) for i, hand in hands_final]
            winners_final = [i for i,s in scores if s==min_score]

            if self.agent_id in winners_final:
                reward -= 0.5  # fold incorrecto
            else:
                reward += 0.5  # fold correcto
        return reward

    def render(self):
        print("=====================================")
        print("Board:")
        Card.print_pretty_cards(self.board)
        print("Hero Hand:")
        Card.print_pretty_cards(self.hands[self.agent_id])
        print("Stacks:", self.stacks)
        print("Pot:", self.pot)
        print("Current Bet:", self.current_bet)
        print("Active Players:", self.active_players)