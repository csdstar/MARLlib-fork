"""
Quick Start 示例：使用 MARLlib 在 MPE simple_spread 环境上训练 MAPPO
适合第一次验证环境和算法是否安装正确
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from marllib import marl
from gym.envs.registration import register
from marllib.envs.base_env import ENV_REGISTRY, RLlibMPE

print("=== Step 1: 创建环境 ===")
# 创建 env_args
env_args = {"map_name": "simple_spread", "force_coop": True}

# 注册一个合法 Gym ID
register(
    id="mpe_simple_spread-v0",
    entry_point=lambda **kwargs: RLlibMPE(ENV_REGISTRY["simple_spread"], env_args)
)

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
    stop={"timesteps_total": 10000},  # 训练少一点即可，快速出结果
    checkpoint_freq=1,
    share_policy="all",
    log_dir="./logs/mappo_simple_spread",
    checkpoint_end=True,
    num_workers=10
)

import glob

print("=== Step 4: 训练完成，开始渲染 ===")
# 自动搜索最新的训练结果文件夹
exp_root = "./exp_results/mappo_mlp_simple_spread"
trial_dirs = sorted(
    glob.glob(os.path.join(exp_root, "MAPPOTrainer_*")),
    key=os.path.getmtime
)
if not trial_dirs:
    raise RuntimeError(f"未找到任何 MAPPOTrainer_* 文件夹，请确认训练是否成功")
latest_trial_dir = trial_dirs[-1]

# 在该 trial 目录中找最新的 checkpoint 文件夹
checkpoint_dirs = sorted(
    glob.glob(os.path.join(latest_trial_dir, "checkpoint_*")),
    key=os.path.getmtime
)
if not checkpoint_dirs:
    raise RuntimeError(f"未找到 checkpoint 文件夹: {latest_trial_dir}")
checkpoint_path = checkpoint_dirs[-1]  # 最新的一个 checkpoint

# params.json 文件路径
params_path = os.path.join(latest_trial_dir, "params.json")

print(f"使用最新的 checkpoint: {checkpoint_path}")
# 直接用 marl 提供的渲染接口
mappo.render(
    env,
    model,
    restore_path={
        "model_path": checkpoint_path,
        "params_path": params_path,
        'render': True  # 是否进行渲染
    },  # 用刚刚训练好的 checkpoint
    video_dir="./logs/mappo_simple_spread/render_video",
    local_mode=True,  # 强制单进程运行，方便渲染
    checkpoint_end=False,  # 不在末尾再保存 checkpoint
    share_policy="all",  # 所有 agent 共享一个策略
    num_gpus=0  # 禁用 GPU，否则找不到就报错
)

print("=== 完成 ===")
print("训练和渲染结果已保存在 ./logs/mappo_simple_spread/render/")
print("可以直接打开 mp4 文件进行播放，或用浏览器打开 gif 查看。")

print("可以用 TensorBoard 可视化：")
print("tensorboard --logdir ./logs/mappo_simple_spread")
