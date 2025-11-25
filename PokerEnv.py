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
    return 2 if eq > 0.20 else 0

def policy_player2(hero_hand, board, num_opponents):
    hero_str = cards_int_to_str(hero_hand)
    eq = estimate_equity(hero_str, board_str=[], num_opponents=num_opponents, iters=2000)['win_prob']
    return random.choice([2,3]) if eq > 0.30 else 0

def policy_player3(hero_hand, board, num_opponents):
    hero_str = cards_int_to_str(hero_hand)
    eq = estimate_equity(hero_str, board_str=[], num_opponents=num_opponents, iters=2000)['win_prob']
    return random.choice([0,2]) if eq > 0.20 else 0

def policy_player4(hero_hand, board, num_opponents):
    hero_str = cards_int_to_str(hero_hand)
    eq = estimate_equity(hero_str, board_str=[], num_opponents=num_opponents, iters=2000)['win_prob']
    if eq > 0.15:
        return 2
    else:
        return 0 if random.random() < 0.7 else 2

# ======================
# Entorno Gymnasium
# ======================

class Poker5EnvFull(Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    ACTIONS = ["Fold", "Check", "Call", "Bet", "Raise"]

    def __init__(self, opponent_policies=None, starting_stack=1000, small_blind=10, big_blind=20):
        super().__init__()

        self.dealer_pos = 0  # posición del botón, rota cada mano
        self.num_players = 5
        self.agent_id = 0
        self.agent_folded = False
        self.starting_stack = starting_stack
        self.preflop_stack = starting_stack
        self.winners = []
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
        self.stacks = [self.starting_stack]*self.num_players
        self.all_in = [False] * self.num_players
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
        for i in range(self.num_players):
            if self.stacks[i] <= 0:
                self.stacks[i] = self.starting_stack
        
        self.deck.shuffle()
        
        self.hands = [self.deck.draw(2) for _ in range(self.num_players)]

        print("Hero Hand:")
        Card.print_pretty_cards(self.hands[self.agent_id])

        self.active_players = [True] * self.num_players
        self.winners = []

        self.bets = [0] * self.num_players
        self.current_bet = self.big_blind
        self.pot = 0

        self.round_stage = 'preflop'
        self.preflop_stack = self.stacks[0]
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
            "current_bet": np.array([self.current_bet], dtype=np.float32),
            "active_players": self.active_players.copy(),
        }

    def step(self, action=None):
        """Step del agente; los oponentes reaccionan internamente"""

        self.done = False
        ronda_done = True
        self.reward = 0

        if self.current_bet == 0 or (self.round_stage == "preflop" and self.pot == self.big_blind + self.small_blind):
            self.render()

        # Determinar orden de turno según dealer
        first = (self.dealer_pos + 2) % self.num_players  # jugador a la derecha de BB
        turn_order = [(first + i) % self.num_players for i in range(self.num_players)]

        for player_id in turn_order:
            if not self.active_players[player_id] or self.done:
                continue

            # Agente
            if player_id == self.agent_id:
                if action is not None:
                    self._apply_action(self.agent_id, action)
                    print(f"Agente tomó acción: {self.ACTIONS[action]}")

            # Oponentes
            else:
                opp_action = self.opponent_policies[player_id-1](self._get_obs())
                self._apply_action(player_id, opp_action)

            if sum(self.all_in) >= 1:
                self.done = True
                break

            # Terminar si solo queda un jugador
            if sum(self.active_players) <= 1:
                self.done = True
                break

        bets_activas = [self.bets[i] for i, active in enumerate(self.active_players) if active]
        ronda_done = len(set(bets_activas)) == 1

        for i in range(self.num_players):
            if self.stacks[i] <= 0:
                ronda_done = True

        # Avanzar ronda
        if not self.done and ronda_done:
            
            self.bets = [0] * self.num_players
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
            elif self.round_stage == 'river':
                self.done = True
        
        if self.done:
            self._resolve_hand()
        else:
            if not self.agent_folded:
                hero_str = cards_int_to_str(self.hands[self.agent_id])
                board_str = cards_int_to_str(self.board)
                equity = estimate_equity(hero_str, board_str=board_str, num_opponents=sum(self.active_players)-1, iters=2000)['win_prob']
                if self.round_stage == 'preflop':
                    if equity > (1 / sum(self.active_players) + 2):
                        self.reward += 10 * equity
                    else:
                        self.reward -= 1 / equity
                elif self.round_stage == 'flop':
                    if equity > (1 / sum(self.active_players) + 1):
                        self.reward += 10 * equity
                    else:
                        self.reward -= 1 / equity
                elif self.round_stage == 'turn':
                    if equity > (1 / sum(self.active_players)):
                        self.reward += 10 * equity
                    else:
                        self.reward -= 1 / equity
                elif self.round_stage == 'river':
                    if equity > (1 / sum(self.active_players)):
                        self.reward += 10 * equity
                    else:
                        self.reward -= 1 / equity

        return self._get_obs(), self.reward, self.done, False, {}

    def get_legal_actions(self, player_id):
        """
        Devuelve una lista de acciones legales (0=fold, 1=check, 2=call, 3=bet, 4=raise)
        teniendo en cuenta la current_bet y el player_bet.
        """
        legal = []
        player_bet = self.bets[player_id]

        if self.agent_folded:
            return legal

        if player_bet == self.current_bet:
            legal.append(1)
        else:
            # --- CALL ---
            legal.append(2)

        # --- BET ---
        # Solo si no hay apuestas previas
        if self.current_bet == 0:
            legal.append(3)

        # --- RAISE ---
        legal.append(4)

        # --- FOLD ---
        # Solo si player_bet < current_bet
        if player_bet < self.current_bet:
            legal.append(0)

        return legal

    def _apply_action(self, player_id, action):
        if not self.active_players[player_id] or self.all_in[player_id]:
            return
        
        player_bet = self.bets[player_id]
        accion_opp = ""
        stack = self.stacks[player_id]

        if action == 0 and player_bet == self.current_bet:
            print(f"Jugador {player_id + 1} tomó acción: Check")
            return

        # Acción fold
        if action == 0:
            accion_opp = "Fold"
            self.active_players[player_id] = False
            if player_id == self.agent_id:
                self.agent_folded = True
                
        # Acción call
        elif action == 2:
            to_call = self.current_bet - player_bet

            if to_call <= 0:
                accion_opp = "Check"
                amount = 0
            
            elif to_call >= stack:
                accion_opp = "Call (All-in)"
                self.all_in[player_id] = True
                amount = stack
            else:
                accion_opp = "Call"
                amount = to_call

            self.stacks[player_id] -= amount
            self.bets[player_id] += amount
            self.pot += amount

        # Acción bet
        elif action == 3:
            if self.current_bet > 0:
                return self._apply_action(player_id, 2)

            if stack <= self.big_blind:
                self.all_in[player_id] = True
                accion_opp = "Bet (All-in)"
                amount = stack
            else:
                accion_opp = "Bet"
                amount = self.big_blind
                

            self.stacks[player_id] -= amount
            self.bets[player_id] += amount
            self.pot += amount
            self.current_bet = self.bets[player_id]

        # Acción raise
        elif action == 4:

            if self.current_bet == 0:
                to_raise = 2 * self.big_blind
            else:
                to_raise = 2 * self.current_bet
            
            if stack + player_bet <= to_raise:
                accion_opp = "Raise (All-in)"
                amount = stack
                self.all_in[player_id] = True
            else:
                accion_opp = "Raise"
                amount = to_raise - player_bet

            self.stacks[player_id] -= amount
            self.bets[player_id] += amount
            self.pot += amount
            
            self.current_bet = self.bets[player_id]
        
        if player_id != 0: 
            print(f"Jugador {player_id + 1} tomó acción: {accion_opp}")

    def _resolve_hand(self):
        active_hands = [(i, self.hands[i]) for i, active in enumerate(self.active_players) if active]

        # Si solo queda un jugador, gana automáticamente
        if len(active_hands) == 1:
            winner_id, winner_hand = active_hands[0]
            self.winners = [winner_id]
            self.stacks[winner_id] += self.pot  # adjudicar pot

        elif sum(self.all_in) > 0:
            self.board += self.deck.draw(5 - len(self.board))
            scores = [(i, self.evaluator.evaluate(hand, self.board)) for i, hand in active_hands]
            min_score = min([s for i,s in scores])
            self.winners = [i for i,s in scores if s==min_score]

            share = self.pot // len(self.winners)

            for w in self.winners:
                self.stacks[w] += share

        else:
            scores = [(i, self.evaluator.evaluate(hand, self.board)) for i, hand in active_hands]
            min_score = min([s for i,s in scores])
            self.winners = [i for i,s in scores if s==min_score]

            share = self.pot // len(self.winners)

            for w in self.winners:
                self.stacks[w] += share

        self.pot = 0 # limpiar el pot

        stack_change = self.stacks[self.agent_id] - self.preflop_stack

        # Equity final
        hero_str = cards_int_to_str(self.hands[self.agent_id])
        board_str = cards_int_to_str(self.board)
        agent_equity_final = estimate_equity(hero_str, board_str=board_str, num_opponents=len(active_hands)-1, iters=2000)['win_prob']
        
        self.reward = stack_change / self.big_blind
        
        # Penaliza menos si perdió con buena mano
        if self.agent_id not in self.winners and stack_change < 0:
            self.reward -= (1 - agent_equity_final)

        # Penaliza o recompensa fold
        if self.agent_folded:
            hands_final = active_hands
            hands_final.append((self.agent_id, self.hands[self.agent_id]))
            self.board += self.deck.draw(5 - len(self.board))
            scores_final = [(i, self.evaluator.evaluate(hand, self.board)) for i, hand in hands_final]
            min_score_final = min([s for i,s in scores_final])
            winners_final = [i for i,s in scores_final if s==min_score_final]

            if self.agent_id in winners_final and len(winners_final) == 1:
                self.reward -= 0.25  # fold incorrecto
            else:
                self.reward += 0.25  # fold correcto

    def render(self):
        print("=====================================")
        print("Board:")
        Card.print_pretty_cards(self.board)
        print("Hero Hand:")
        Card.print_pretty_cards(self.hands[self.agent_id])
        print("\nStacks:", self.stacks)
        print("Pot:", self.pot)
        print("Current Bet:", self.current_bet)
        print("Active Players:", self.active_players)
        print("\n")