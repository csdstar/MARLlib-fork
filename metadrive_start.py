import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from marllib import marl

print("=== Step 1: 创建环境 ===")
# 改动1：环境改为 metadrive
env = marl.make_env(environment_name="metadrive", map_name="Bottleneck")

print("=== Step 2: 初始化MAPPO算法 ===")
# 改动2：超参数源改为 metadrive
# mappo = marl.algos.mappo(hyperparam_source="metadrive")
mappo = marl.algos.mappo(hyperparam_source="mappo_metadrive")

print("=== Step 2.5: 构建模型 ===")
model_preference = {"core_arch": "mlp", "encode_layer": "128-256"}
model = marl.build_model(env, mappo, model_preference)

print("=== Step 3: 开始训练 ===")
trainer = mappo.fit(
    env,
    model,
    stop={"timesteps_total": 20000},  # MetaDrive 环境较重，可先少跑
    checkpoint_freq=1,
    share_policy="all",
    log_dir="./logs/mappo_metadrive",
    checkpoint_end=True,
    num_workers=8  # 建议减少并发以防内存占用过大
)

print("=== 完成 ===")
print("可视化命令：tensorboard --logdir ./logs/mappo_metadrive")
