"""
Quick Start 示例：使用 MARLlib 在 MPE simple_spread 环境上训练 MAPPO
适合第一次验证环境和算法是否安装正确
"""

import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from marl import marl

# 1. 创建环境
# mpe = Multi-Agent Particle Environment，simple_spread 是最经典的协作任务
print("=== Step 1: 创建环境 ===")
env = marl.make_env(environment_name="mpe", map_name="simple_spread")

# 2. 初始化算法
# MAPPO = Multi-Agent PPO，适合协作任务
print("=== Step 2: 初始化MAPPO算法 ===")
mappo = marl.algos.mappo(hyperparam_source="mpe")

# 3. 训练
# stop 参数指定训练总步数，这里设置 5万步，可以适当增大
# share_policy='group' 表示共享策略，提高样本效率
print("=== Step 3: 开始训练 ===")
mappo.fit(
    env,
    stop={'timesteps_total': 50000},
    share_policy='group',
    log_dir="./logs/mappo_simple_spread"
)

print("=== Step 4: 训练完成，开始评估 ===")
# 4. 评估
# 如果你有 GUI/X11，可以把 render=True 看到小球移动
# 在纯 WSL 可以先设置 render=False 只看 reward
mappo.evaluate(env, n_eval_episodes=5, render=False)

print("训练和评估结束，日志已保存在 ./logs/mappo_simple_spread")
print("可以用 TensorBoard 可视化：")
print("tensorboard --logdir ./logs/mappo_simple_spread")
