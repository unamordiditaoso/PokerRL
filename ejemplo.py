from PokerEnv import Poker5EnvFull
from treys import Card
import random

# Crear el entorno
env = Poker5EnvFull()

# Reiniciar entorno
obs, _ = env.reset()
done = False
step_count = 0

initial_stack = env.stacks[0]

# === Demo: 10 manos con blinds rotativos y steps visibles ===

num_hands = 10

# Guardar estado inicial del deck y del board
initial_deck_state = env.deck.cards.copy()
initial_board = env.board.copy()

for hand_idx in range(num_hands):
    print(f"\n=== MANO {hand_idx+1} ===")

    # Reset parcial de la mano con blinds rotativos
    obs = env.partial_reset()  # repartirá nuevas manos y asignará SB/BB
    done = False
    step_count = 0
    initial_stack = env.stacks[env.agent_id]

    while not done:
        step_count += 1

        # Acción del agente (ejemplo aleatoria, reemplaza por tu RL)
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)

        print(f"\n--- Step {step_count} ---")
        print(f"Agente tomó acción: {env.ACTIONS[action]}")
        env.render()  # muestra estado actual (board, bets, stacks)

    final_stack = env.stacks[env.agent_id]
    net_gain = final_stack - initial_stack

    # Mostrar board bonito
    print("\n=== CARTAS EN EL TABLERO ===")
    Card.print_pretty_cards(env.board)

    # Mostrar manos de todos los jugadores
    active_hands = [(i, env.hands[i]) for i, active in enumerate(env.active_players) if active]
    print("\n=== MANOS DE LOS JUGADORES ===")
    for i, hand in active_hands:
        name = "Agente" if i == env.agent_id else f"Jugador {i+1}"
        print(f"{name}: ", end="")
        Card.print_pretty_cards(hand)

    # Mostrar ganador(es)
    winner_names = ["Agente" if w == env.agent_id else f"Jugador {w+1}" for w in env.winners]
    print("\n=== GANADOR(ES) ===")
    print(", ".join(winner_names))

    # Mostrar reward y stacks
    print("\n=== FIN DE LA MANO ===")
    print(f"Stack inicial del agente: {initial_stack}")
    print(f"Stack final del agente: {final_stack}")
    print(f"Fichas ganadas/perdidas: {net_gain}")
    print(f"Reward final del agente: {reward}")

    # Restaurar board y deck para la próxima mano
    env.board = initial_board.copy()
    env.deck.cards = initial_deck_state.copy()

print("=====================================")
print("=====================================")