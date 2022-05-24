from stable_baselines3.common.env_checker import check_env
from inventory_control_problem import inventoryControlEnvironment

env = inventoryControlEnvironment()
check_env(env)
print("Environment checked!")