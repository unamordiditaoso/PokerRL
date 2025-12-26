import pygame
from PokerEnv import Poker5EnvFull

# --- Inicializar entorno ---
env = Poker5EnvFull()
obs = env.partial_reset()

# --- Pygame ---
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("PokerRL Simulator")
clock = pygame.time.Clock()

# --- Constantes ---
CARD_SIZE = (80, 120)
RANKS = "23456789TJQKA"
SUITS = "cdhs"

# --- Posiciones jugadores ---
PLAYER_POSITIONS = [
    (175, 450),  # Hero
    (75, 100),  # Jugador 1
    (325, 50),   # Jugador 2
    (575, 100),  # Jugador 3
    (475, 450),  # Jugador 4
]

# --- Cartas comunitarias ---
BOARD_POSITION = (175, 250)
BOARD_SPACING = 100

# --- Cargar imágenes ---
card_images = {}
BACK_CARD = pygame.image.load("cartas/back.png").convert_alpha()
BACK_CARD = pygame.transform.scale(BACK_CARD, CARD_SIZE)

def card_idx_to_str(idx):
    rank = RANKS[idx % 13]
    suit = SUITS[idx // 13]
    return rank + suit

def cards_idx_to_str(cards_idx):
    return [card_idx_to_str(c) for c in cards_idx]

def load_card(card_str):
    if card_str not in card_images:
        img = pygame.image.load(f"cartas/{card_str}.png").convert_alpha()
        card_images[card_str] = pygame.transform.scale(img, CARD_SIZE)
    return card_images[card_str]

# --- Loop principal ---
running = True
while running:
    clock.tick(30)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        # Avanzar el entorno con tecla SPACE (ejemplo)
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            action = 0  # Fold / Check / Dummy
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.partial_reset()

    # --- Dibujar mesa ---
    screen.fill((0, 128, 0))  # color verde

    # --- Dibujar jugadores ---
    for i, pos in enumerate(PLAYER_POSITIONS):
        x, y = pos
        if i == 0:  # Hero
            hero_hand_idx = obs["hero_hand"]
            hero_hand_str = cards_idx_to_str(hero_hand_idx)
            for j, card in enumerate(hero_hand_str):
                screen.blit(load_card(card), (x + j*100, y))
        else:  # Oponentes
            for j in range(2):
                screen.blit(BACK_CARD, (x + j*100, y))

    # --- Dibujar cartas comunitarias ---
    board_idx = obs.get("board", [])  # lista de 5 elementos, None si no hay carta
    
    for j, idx in enumerate(board_idx):
        x = BOARD_POSITION[0] + j*BOARD_SPACING
        y = BOARD_POSITION[1]

        if idx is not None and idx != 52:  # 52 = valor reservado para "no hay carta"
            card_str = card_idx_to_str(idx)
            screen.blit(load_card(card_str), (x, y))
        else:
            screen.blit(BACK_CARD, (x, y))

    pygame.display.flip()

pygame.quit()
