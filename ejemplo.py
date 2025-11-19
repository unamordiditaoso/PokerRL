from PokerEnv import Poker5EnvFull
from treys import Card
import random

# Crear el entorno
env = Poker5EnvFull()

# Reiniciar entorno
obs, _ = env.reset()
done = False
step_count = 0

print("\n=== INICIO DE LA MANO ===")
env.render()

initial_stack = env.stacks[0]

# Demo loop: el agente hace acciones aleatorias (luego se reemplaza por RL)
while not done:
    step_count += 1
    # Acción del agente: fold/call/bet/raise aleatorio
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)

    print(f"\n--- Step {step_count} ---")
    print(f"Agente tomó acción: {env.ACTIONS[action]}")
    env.render()

final_stack = env.stacks[0]
net_gain = final_stack - initial_stack

# Crear hands_final incluyendo agente incluso si foldeó
hands_final = [(i, hand) for i, hand in enumerate(env.hands) if env.active_players[i]]  # jugadores activos
# Añadir agente si foldeó
if not env.active_players[env.agent_id]:
    hands_final.append((env.agent_id, env.hands[env.agent_id]))

# Mostrar las manos de todos los jugadores al final
print("\n=== CARTAS DE TODOS LOS JUGADORES AL FINAL ===")
for i, hand in hands_final:
    hand_str = [Card.int_to_str(c) for c in hand]  # convertir a 'As','Kd', etc.
    if i == env.agent_id:
        print(f"Agente: {hand_str}")
    else:
        print(f"Jugador {i+1}: {hand_str}")


print("\n=== FIN DE LA MANO ===")
print(f"Stack inicial del agente: {initial_stack}")
print(f"Stack final del agente: {final_stack}")
print(f"Fichas ganadas/perdidas: {net_gain}")
print(f"Reward final del agente: {reward}")

print("=====================================")
print("=====================================")