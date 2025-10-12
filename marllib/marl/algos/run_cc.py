# MIT License

# Copyright (c) 2023 Replicable-MARL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import gym
import ray
from ray import tune
from ray.rllib.utils.framework import try_import_torch

from marllib.marl.algos.scripts import POlICY_REGISTRY
from marllib.marl.common import recursive_dict_update, dict_update

torch, nn = try_import_torch()


def restore_config_update(exp_info, run_config, stop_config):
    if exp_info['restore_path']['model_path'] == '':
        restore_config = None
    else:
        restore_config = exp_info['restore_path']
        if 'render' in exp_info['restore_path']:
            render_config = {
                "evaluation_interval": 1,
                "evaluation_num_episodes": 100,
                "evaluation_num_workers": 1,
                "evaluation_config": {
                    "record_env": False,
                    "render_env": True,
                }
            }

            run_config = recursive_dict_update(run_config, render_config)

            render_stop_config = {
                "training_iteration": 1,
            }

            stop_config = recursive_dict_update(stop_config, render_stop_config)

    return exp_info, run_config, stop_config, restore_config


def run_cc(exp_info, env, model, stop=None):
    """
   运行集中式评价（Central Critic）类算法的训练流程。
   参数:
       exp_info: 实验信息字典，包含算法名、环境名、超参数、训练配置等
       env: 由 make_env() 创建的多智能体环境实例
       model: 模型配置（通常是网络结构定义）
       stop: 可选的训练终止条件（覆盖默认条件）

   返回:
       results: 训练结果（Ray Tune 的返回对象）
   """

    # 初始化ray
    ray.init(local_mode=exp_info["local_mode"], num_gpus=exp_info["num_gpus"])

    ########################
    ### environment info ###
    ########################
    # 从环境实例中提取元信息（观测空间、动作空间、智能体数量等）
    env_info = env.get_env_info()
    map_name = exp_info['env_args']['map_name']
    agent_name_ls = env.agents
    env_info["agent_name_ls"] = agent_name_ls
    env.close()

    ######################
    ### space checking ###
    ######################

    # 检查动作空间类型（离散 / 连续）
    action_discrete = isinstance(env_info["space_act"], gym.spaces.Discrete) or isinstance(env_info["space_act"], gym.spaces.MultiDiscrete)
    # 离散动作空间下不允许使用连续型算法 maddpg
    if action_discrete:
        if exp_info["algorithm"] in ["maddpg"]:
            raise ValueError(
                "Algo -maddpg- only supports continuous action space, Env -{}- requires Discrete action space".format(
                    exp_info["env"]))
    # 连续动作空间下不允许使用离散型算法 coma
    else:
        if exp_info["algorithm"] in ["coma"]:
            raise ValueError(
                "Algo -coma- only supports discrete action space, Env -{}- requires continuous action space".format(
                    exp_info["env"]))

    ######################
    ### policy sharing ###
    ######################

    # 获取环境提供的智能体策略映射信息
    policy_mapping_info = env_info["policy_mapping_info"]

    # 部分环境为多场景结构，因此需根据 map_name 选择具体配置
    if "all_scenario" in policy_mapping_info:
        policy_mapping_info = policy_mapping_info["all_scenario"]
    else:
        policy_mapping_info = policy_mapping_info[map_name]

    # 设置共享策略名：
    # 若 agent_level_batch_update=True，则每个 agent 独立更新 -> default_policy
    # 否则共享同一参数 -> shared_policy
    shared_policy_name = "default_policy" if exp_info["agent_level_batch_update"] else "shared_policy"

    # ====== 策略共享模式判断 ======
    if exp_info["share_policy"] == "all":
        if not policy_mapping_info["all_agents_one_policy"]:
            raise ValueError("in {}, policy can not be shared, change it to 1. group 2. individual".format(map_name))
        policies = {shared_policy_name}
        policy_mapping_fn = (
            lambda agent_id, episode, **kwargs: shared_policy_name)

    elif exp_info["share_policy"] == "group":
        groups = policy_mapping_info["team_prefix"]

        if len(groups) == 1:
            if not policy_mapping_info["all_agents_one_policy"]:
                raise ValueError(
                    "in {}, policy can not be shared, change it to 1. group 2. individual".format(map_name))

            policies = {shared_policy_name}
            policy_mapping_fn = (
                lambda agent_id, episode, **kwargs: shared_policy_name)

        else:
            policies = {
                "policy_{}".format(i): (None, env_info["space_obs"], env_info["space_act"], {}) for i in
                groups
            }
            policy_ids = list(policies.keys())
            policy_mapping_fn = tune.function(
                lambda agent_id: "policy_{}_".format(agent_id.split("_")[0]))

    elif exp_info["share_policy"] == "individual":
        if not policy_mapping_info["one_agent_one_policy"]:
            raise ValueError("in {}, agent number too large, we disable no sharing function".format(map_name))

        policies = {
            "policy_{}".format(i): (None, env_info["space_obs"], env_info["space_act"], {}) for i in
            range(env_info["num_agents"])
        }
        policy_ids = list(policies.keys())
        policy_mapping_fn = tune.function(
            lambda agent_id: policy_ids[agent_name_ls.index(agent_id)])

    else:
        raise ValueError("wrong share_policy {}".format(exp_info["share_policy"]))

    # ====== 特殊算法强制独立策略 ======
    if exp_info["algorithm"] in ["happo", "hatrpo"]:
        if not policy_mapping_info["one_agent_one_policy"]:
            raise ValueError("in {}, agent number too large, we disable no sharing function".format(map_name))

        policies = {
            "policy_{}".format(i): (None, env_info["space_obs"], env_info["space_act"], {}) for i in
            range(env_info["num_agents"])
        }
        policy_ids = list(policies.keys())
        policy_mapping_fn = tune.function(
            lambda agent_id: policy_ids[agent_name_ls.index(agent_id)])

    #########################
    ### experiment config ###
    #########################

    # 构造 Ray Tune 的运行配置
    run_config = {
        "seed": int(exp_info["seed"]),
        "env": exp_info["env"] + "_" + exp_info["env_args"]["map_name"],
        "num_gpus_per_worker": exp_info["num_gpus_per_worker"],
        "num_gpus": exp_info["num_gpus"],
        "num_workers": exp_info["num_workers"],
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn
        },
        "framework": exp_info["framework"],
        "evaluation_interval": exp_info["evaluation_interval"],
        "simple_optimizer": False  # force using better optimizer
    }

    # 训练停止条件
    stop_config = {
        "episode_reward_mean": exp_info["stop_reward"],
        "timesteps_total": exp_info["stop_timesteps"],
        "training_iteration": exp_info["stop_iters"],
    }

    stop_config = dict_update(stop_config, stop)

    exp_info, run_config, stop_config, restore_config = restore_config_update(exp_info, run_config, stop_config)

    ##################
    ### run script ###
    ##################

    # 调用对应算法模块（从 POLICY_REGISTRY 中查找，如 mappo, maddpg, qmix 等）
    results = POlICY_REGISTRY[exp_info["algorithm"]](model, exp_info, run_config, env_info, stop_config, restore_config)

    ray.shutdown()

    return results
