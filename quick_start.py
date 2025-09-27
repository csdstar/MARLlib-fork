"""
Quick Start 示例：使用 MARLlib 在 MPE simple_spread 环境上训练 MAPPO
适合第一次验证环境和算法是否安装正确
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from marllib import marl

print("=== Step 1: 创建环境 ===")
env = marl.make_env(environment_name="mpe", map_name="simple_spread")

print("=== Step 2: 初始化MAPPO算法 ===")
mappo = marl.algos.mappo(hyperparam_source="mpe")

print("=== Step 2.5: 构建模型 ===")
model_preference = {"core_arch": "mlp", "encode_layer": "128-256"}
model_class, model_config = marl.build_model(env, mappo, model_preference)

print("=== Step 3: 开始训练 ===")
model = (model_class, model_config)
trainer = mappo.fit(
    env,
    model,
    stop={"timesteps_total": 10000},  # 训练少一点即可，快速出结果
    checkpoint_freq=1,
    share_policy="all",
    log_dir="./logs/mappo_simple_spread",
    checkpoint_end=True,
    num_workers=2
)

print("=== Step 4: 训练完成，开始渲染 ===")
# 直接用 marl 提供的渲染接口
# render() 会创建视频文件保存到 log_dir/render 文件夹
mappo.render(
    env,
    model,
    restore_path=trainer,  # 用刚刚训练好的 checkpoint
    render_num=1,  # 渲染1个 episode
    save_gif=True,  # 保存为 gif/mp4
    save_dir="./logs/mappo_simple_spread/render",
    fps=30,
    local_mode=True,  # 强制单进程运行，方便渲染
    num_gpus=0  # 禁用 GPU，否则找不到就报错
)

print("=== 完成 ===")
print("训练和渲染结果已保存在 ./logs/mappo_simple_spread/render/")
print("可以直接打开 mp4 文件进行播放，或用浏览器打开 gif 查看。")

print("可以用 TensorBoard 可视化：")
print("tensorboard --logdir ./logs/mappo_simple_spread")
