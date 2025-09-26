# quick_start_debug.py
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 打开 faulthandler 用于在 C 层崩溃时尽可能打印信息
import faulthandler
faulthandler.enable()

# 限制线程数（二次保险）
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# 可选：降低 ray 日志噪音
os.environ.setdefault("RAY_IGNORE_UNHANDLED_ERRORS", "1")

from marllib import marl

print("=== 创建环境 ===")
env = marl.make_env(environment_name="mpe", map_name="simple_spread")

print("=== 初始化 MAPPO ===")
mappo = marl.algos.mappo(hyperparam_source="mpe")

print("=== 构建模型偏好 ===")
model_pref = {"core_arch": "mlp", "encode_layer": "128-128"}
model_class, model_config = marl.build_model(env, mappo, model_pref)
model = (model_class, model_config)

print("=== 准备开始训练（local_mode / 单 worker） ===")
# 下面把 local_mode=True 传入 fit()，并限制 num_workers=0（确保不启多进程）
mappo.fit(
    env,
    model,
    stop={"timesteps_total": 10000},
    local_mode=True,        # 在 driver 进程单线程运行（用于调试）
    num_workers=0,          # 不启 worker，多半可避免 SIGABRT
    share_policy='group',
    log_dir="./logs/mappo_debug"
)

print("训练结束（debug 模式）")
