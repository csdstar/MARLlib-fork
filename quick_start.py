"""
Quick Start 示例：使用 MARLlib 在 MPE simple_spread 环境上训练 MAPPO
适合第一次验证环境和算法是否安装正确
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from marllib import marl
from marllib.marl.algos.utils.rollout import rollout

print("=== Step 1: 创建环境 ===")
env = marl.make_env(environment_name="mpe", map_name="simple_spread")

print("=== Step 2: 初始化MAPPO算法 ===")
mappo = marl.algos.mappo(hyperparam_source="mpe")

print("=== Step 2.5: 构建模型 ===")
# 定义一个简单的模型偏好
model_preference = {
    "core_arch": "mlp",      # 使用MLP网络
    "encode_layer": "128-128"
}
model_class, model_config = marl.build_model(env, mappo, model_preference)

print("=== Step 3: 开始训练 ===")
model = (model_class, model_config)
mappo.fit(
    env,
    model,
    stop={'timesteps_total': 50000},
    share_policy='group',
    log_dir="./logs/mappo_simple_spread"
)

print("=== Step 4: 训练完成，开始评估 ===")
mappo.evaluate(env, model, n_eval_episodes=5, render=False)

print("训练和评估结束，日志已保存在 ./logs/mappo_simple_spread")
print("可以用 TensorBoard 可视化：")
print("tensorboard --logdir ./logs/mappo_simple_spread")