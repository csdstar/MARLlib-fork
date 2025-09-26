"""
example of how to render a trajectory on MPE
"""

from marllib import marl

# prepare the environment
env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True)

# initialize algorithm and load hyperparameters
mappo = marl.algos.mappo(hyperparam_source="mpe")

# build agent model based on env + algorithms + user preference if checked available
model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})

# rendering
mappo.render(
    env,
    model,
    restore_path={
        'params_path': "examples/checkpoint/params.json",  # 训练超参数
        'model_path': "examples/checkpoint/checkpoint-6250",  # 训练好的模型检查点路径
        'render': False      # 是否进行渲染
    },
    local_mode=True,        # 用单进程运行（方便调试和渲染）
    share_policy="all",     # 所有 agent 共享一个策略
    checkpoint_end=False,   # 不在末尾再保存 checkpoint
    video_dir="./render_videos"
)
