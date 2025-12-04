import os
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import DummyVecEnv
from PokerEnv import Poker5EnvFull  # tu entorno

# ===========================
# Parámetros de entrenamiento
# ===========================
SEED = 42
TOTAL_TIMESTEPS = 2_000_000
EVAL_FREQ = 5000
N_EVAL_EPISODES = 5
REWARD_TARGET = 500  # ajusta según tu entorno
CHECKPOINT_DIR = "./checkpoints_poker/"
TENSORBOARD_DIR = "./tensorboard_poker/"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)

# ===========================
# Wrapper para partial reset
# ===========================
class PokerPartialResetWrapper(gym.Wrapper):
    """
    Wrapper que:
    - Hace que reset() sea solo partial_reset()
    - Compatible con SB3 DummyVecEnv
    """
    def reset(self, *, seed=None, options=None):
        obs = self.env.partial_reset()  # reset de la mano
        return obs, {}  # devuelve obs e info vacío

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

# ===========================
# Crear entornos
# ===========================
def make_env():
    env = Poker5EnvFull()
    env = PokerPartialResetWrapper(env)
    return env

env = DummyVecEnv([make_env])
eval_env = DummyVecEnv([make_env])

# ===========================
# Callback para guardar mejor modelo
# ===========================
callback_on_best = StopTrainingOnRewardThreshold(REWARD_TARGET, verbose=1)
eval_callback = EvalCallback(
    eval_env,
    callback_on_new_best=callback_on_best,
    eval_freq=EVAL_FREQ,
    n_eval_episodes=N_EVAL_EPISODES,
    best_model_save_path=CHECKPOINT_DIR,
    deterministic=True,
    render=False,
)

# ===========================
# Crear el agente PPO
# ===========================
# Si tu observación es un dict, usar MultiInputPolicy
obs_space = env.observation_space
policy_type = "MultiInputPolicy" if isinstance(obs_space, gym.spaces.Dict) else "MlpPolicy"

model = PPO(
    policy=policy_type,
    env=env,
    verbose=1,
    seed=SEED,
    tensorboard_log=TENSORBOARD_DIR,
    learning_rate=2.5e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
)

# ===========================
# Entrenar el agente
# ===========================
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=eval_callback,
    tb_log_name="PPO_Poker_Run",
)

# ===========================
# Guardar modelo final
# ===========================
model.save(os.path.join(CHECKPOINT_DIR, "ppo_poker_final"))

print("✅ Entrenamiento completado. Mejor modelo guardado en:", CHECKPOINT_DIR)