"""
Quick Start 示例：使用 MARLlib 在 MPE simple_spread 环境上训练 MAPPO
适合第一次验证环境和算法是否安装正确 conda activate py38
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from marllib import marl
from ray.tune.registry import register_env

print("=== Step 1: 创建环境 ===")
env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True)

print("=== Step 2: 初始化MAPPO算法 ===")
mappo = marl.algos.mappo(hyperparam_source="mpe")

print("=== Step 2.5: 构建模型 ===")
model_preference = {"core_arch": "mlp", "encode_layer": "128-256"}
model = marl.build_model(env, mappo, model_preference)

print("=== Step 3: 开始训练 ===")
trainer = mappo.fit(
    env,
    model,
    stop={"timesteps_total": 50000},  # 训练少一点即可，快速出结果
    checkpoint_freq=1,
    share_policy="all",
    log_dir="./logs/mappo_simple_spread",
    checkpoint_end=True,
    num_workers=10
)

print("=== 完成 ===")
print("可以用 TensorBoard 可视化：")
print("tensorboard --logdir exp_results/mappo_mlp_simple_spread/")
