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

from marllib.marl.common import dict_update, get_model_config, check_algo_type, \
    recursive_dict_update
from marllib.marl.algos import run_il, run_vd, run_cc
from marllib.marl.algos.scripts import POlICY_REGISTRY
from marllib.envs.base_env import ENV_REGISTRY
from marllib.envs.global_reward_env import COOP_ENV_REGISTRY
from marllib.marl.models import BaseRNN, BaseMLP, CentralizedCriticRNN, CentralizedCriticMLP, ValueDecompRNN, \
    ValueDecompMLP, JointQMLP, JointQRNN, DDPGSeriesRNN, DDPGSeriesMLP
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune import register_env
from copy import deepcopy
from tabulate import tabulate
from typing import Any, Dict, Tuple
import yaml
import os
import sys

SYSPARAMs = deepcopy(sys.argv)


def set_ray(config: Dict):
    """
    function of combining ray config with other configs
    :param config: dictionary of config to be combined with
    """
    # default config
    with open(os.path.join(os.path.dirname(__file__), "ray/ray.yaml"), "r") as f:
        ray_config_dict = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

    # user config
    user_ray_args = {}
    for param in SYSPARAMs:
        if param.startswith("--ray_args"):
            if "=" in param:
                key, value = param.split(".")[1].split("=")
                user_ray_args[key] = value
            else:  # local_mode
                user_ray_args[param.split(".")[1]] = True

    # update config
    ray_config_dict = dict_update(ray_config_dict, user_ray_args, True)

    for key, value in ray_config_dict.items():
        config[key] = value

    return config


def make_env(
        environment_name: str,
        map_name: str,
        force_coop: bool = False,
        abs_path: str = "",
        **env_params
) -> Tuple[MultiAgentEnv, Dict]:
    """
    创建环境
    construct the environment and register.
    Args:
        :param environment_name: name of the environment
        环境名
        :param map_name: name of the scenario
        场景名
        :param force_coop: enforce the reward return of the environment to be global
        是否强制启用“全局奖励模式”，用于集中式训练
        :param abs_path: env configuration path
        指定环境配置文件的绝对路径
        :param env_params: parameters that can be pass to the environment for customizing the environment
        运行时传入的额外环境参数

    Returns:
        Tuple[MultiAgentEnv, Dict]: env instance & env configuration dict
    """
    # 确定配置文件路径
    if abs_path != "":
        # 用户自定义路径
        env_config_file_path = os.path.join(os.path.dirname(__file__), abs_path)
    else:
        # 默认路径：envs/base_env/config/<environment_name>.yaml
        env_config_file_path = os.path.join(
            os.path.dirname(__file__),
            f"../envs/base_env/config/{environment_name}.yaml"
        )

    # 加载 YAML 配置文件
    with open(env_config_file_path, "r") as f:
        env_config_dict = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

    # 更新配置中 env_args 参数（合并默认配置与用户传入的 env_params）
    # dict_update 是框架内定义的工具函数，支持递归合并字典
    env_config_dict["env_args"] = dict_update(env_config_dict["env_args"], env_params, True)

    # 从命令行中解析以 "--env_args.xxx=yyy" 形式传入的参数
    # SYSPARAMs 是全局变量，记录启动时命令行参数
    user_env_args = {}
    for param in SYSPARAMs:
        if param.startswith("--env_args"):
            key, value = param.split(".")[1].split("=")
            user_env_args[key] = value

    # 将命令行参数合并进配置
    env_config_dict["env_args"] = dict_update(env_config_dict["env_args"], user_env_args, True)

    # 添加场景名和是否强制全局奖励标志
    env_config_dict["env_args"]["map_name"] = map_name
    env_config_dict["force_coop"] = force_coop

    # 统一转换为 Ray 训练格式（如添加 num_workers、rollout_length 等参数）
    env_config = set_ray(env_config_dict)

    # 检查该环境是否已在 ENV_REGISTRY 中注册
    env_reg_ls = []
    check_current_used_env_flag = False
    for env_n in ENV_REGISTRY.keys():
        if isinstance(ENV_REGISTRY[env_n], str):
            # 若值是字符串，说明注册失败（可能是错误信息）
            info = [env_n, "Error", ENV_REGISTRY[env_n],
                    f"envs/base_env/config/{env_n}.yaml",
                    f"envs/base_env/{env_n}.py"]
            env_reg_ls.append(info)
        else:
            # 正常注册的环境
            info = [env_n, "Ready", "Null",
                    f"envs/base_env/config/{env_n}.yaml",
                    f"envs/base_env/{env_n}.py"]
            env_reg_ls.append(info)
            if env_n == env_config["env"]:
                check_current_used_env_flag = True

    # 打印当前环境注册状态表
    print(tabulate(env_reg_ls,
                   headers=['Env_Name', 'Check_Status', "Error_Log", "Config_File_Location", "Env_File_Location"],
                   tablefmt='grid'))

    # 若环境未被正确注册则报错退出
    if not check_current_used_env_flag:
        raise ValueError(
            f"environment \"{env_config['env']}\" not installed properly or not registered yet, "
            f"please see the Error_Log below"
        )

    # 为当前环境生成注册名（如 "mpe_simple_spread"）
    env_reg_name = env_config["env"] + "_" + env_config["env_args"]["map_name"]

    # 根据 force_coop 决定使用哪种注册表（普通 / 全局奖励）
    if env_config["force_coop"]:
        # 全局奖励环境注册
        register_env(env_reg_name,
                     lambda _: COOP_ENV_REGISTRY[env_config["env"]](env_config["env_args"]))
        env = COOP_ENV_REGISTRY[env_config["env"]](env_config["env_args"])
    else:
        # 普通环境注册
        register_env(env_reg_name,
                     lambda _: ENV_REGISTRY[env_config["env"]](env_config["env_args"]))
        env = ENV_REGISTRY[env_config["env"]](env_config["env_args"])

    # 返回环境实例与配置字典
    return env, env_config


def build_model(
        environment: Tuple[MultiAgentEnv, Dict],
        algorithm: str,
        model_preference: Dict,
) -> Tuple[Any, Dict]:
    """
    construct the model
    Args:
        :param environment: name of the environment
        :param algorithm: name of the algorithm
        :param model_preference:  parameters that can be pass to the model for customizing the model

    Returns:
        Tuple[Any, Dict]: model class & model configuration
    """

    if algorithm.name in ["iddpg", "facmac", "maddpg"]:
        if model_preference["core_arch"] in ["gru", "lstm"]:
            model_class = DDPGSeriesRNN
        else:
            model_class = DDPGSeriesMLP

    elif algorithm.name in ["qmix", "vdn", "iql"]:
        if model_preference["core_arch"] in ["gru", "lstm"]:
            model_class = JointQRNN
        else:
            model_class = JointQMLP

    else:
        if algorithm.algo_type == "IL":
            if model_preference["core_arch"] in ["gru", "lstm"]:
                model_class = BaseRNN
            else:
                model_class = BaseMLP
        elif algorithm.algo_type == "CC":
            if model_preference["core_arch"] in ["gru", "lstm"]:
                model_class = CentralizedCriticRNN
            else:
                model_class = CentralizedCriticMLP
        else:  # VD
            if model_preference["core_arch"] in ["gru", "lstm"]:
                model_class = ValueDecompRNN
            else:
                model_class = ValueDecompMLP

    if model_preference["core_arch"] in ["gru", "lstm"]:
        model_config = get_model_config("rnn")
    elif model_preference["core_arch"] in ["mlp"]:
        model_config = get_model_config("mlp")
    else:
        raise NotImplementedError("{} not supported agent model arch".format(model_preference["core_arch"]))

    if len(environment[0].observation_space.spaces["obs"].shape) == 1:
        encoder = "fc_encoder"
    else:
        encoder = "cnn_encoder"

    # encoder config
    encoder_arch_config = get_model_config(encoder)
    model_config = recursive_dict_update(model_config, encoder_arch_config)
    model_config = recursive_dict_update(model_config, {"model_arch_args": model_preference})

    if algorithm.algo_type == "VD":
        mixer_arch_config = get_model_config("mixer")
        model_config = recursive_dict_update(model_config, mixer_arch_config)
        if "mixer_arch" in model_preference:
            recursive_dict_update(model_config, {"model_arch_args": model_preference})

    return model_class, model_config


class _Algo:
    """An algorithm tool class
    :param str algo_name: the algorithm name
    """

    def __init__(self, algo_name: str):

        if "_" in algo_name:
            self.name = algo_name.split("_")[0].lower()
            self.algo_type = algo_name.split("_")[1].upper()
        else:
            self.name = algo_name
            self.algo_type = check_algo_type(self.name.lower())
        self.algo_parameters = {}
        self.config_dict = None
        self.common_config = None

    def __call__(self, hyperparam_source: str, **algo_params):
        """
        Args:
            :param hyperparam_source: source of the algorithm's hyperparameter
            options:
            1. "common" use config under "marl/algos/hyperparams/common"
            2. $environment use config under "marl/algos/hyperparams/finetuned/$environment"
            3. "test" use config under "marl/algos/hyperparams/test"
        Returns:
            _Algo
        """
        if hyperparam_source in ["common", "test"]:
            rel_path = "algos/hyperparams/{}/{}.yaml".format(hyperparam_source, self.name)
        else:
            rel_path = "algos/hyperparams/finetuned/{}/{}.yaml".format(hyperparam_source, self.name)

        if not os.path.exists(os.path.join(os.path.dirname(__file__), rel_path)):
            rel_path = "../../examples/config/algo_config/{}.yaml".format(self.name)

        with open(os.path.join(os.path.dirname(__file__), rel_path), "r") as f:
            algo_config_dict = yaml.load(f, Loader=yaml.FullLoader)
            f.close()

        # update function-fixed config
        algo_config_dict["algo_args"] = dict_update(algo_config_dict["algo_args"],
                                                    algo_params, True)

        # user config
        user_algo_args = {}
        for param in SYSPARAMs:
            if param.startswith("--algo_args"):
                value = param.split("=")[1]
                key = param.split("=")[0].split(".")[1]
                user_algo_args[key] = value

        # update commandline config
        algo_config_dict["algo_args"] = dict_update(algo_config_dict["algo_args"],
                                                    user_algo_args, True)

        self.algo_parameters = algo_config_dict

        return self

    def fit(self, env: Tuple[MultiAgentEnv, Dict], model: Tuple[Any, Dict], stop: Dict = None,
            **running_params) -> None:
        """
        Entering point of the whole training
        Args:
            :param env: a tuple of environment instance and environmental configuration
            :param model: a tuple of model class and model configuration
            :param stop: dict of running stop condition
            :param running_params: other configuration to customize the training
        Returns:
            None
        """

        env_instance, info = env
        model_class, model_info = model

        self.config_dict = info
        self.config_dict = recursive_dict_update(self.config_dict, model_info)

        self.config_dict = recursive_dict_update(self.config_dict, self.algo_parameters)
        self.config_dict = recursive_dict_update(self.config_dict, running_params)

        self.config_dict['algorithm'] = self.name

        if self.algo_type == "IL":
            return run_il(self.config_dict, env_instance, model_class, stop=stop)
        elif self.algo_type == "VD":
            return run_vd(self.config_dict, env_instance, model_class, stop=stop)
        elif self.algo_type == "CC":
            return run_cc(self.config_dict, env_instance, model_class, stop=stop)
        else:
            raise ValueError("not supported type {}".format(self.algo_type))

    def render(self, env: Tuple[MultiAgentEnv, Dict], model: Tuple[Any, Dict], stop: Dict = None,
               **running_params) -> None:
        """
        Entering point of the rendering, running a one iteration fit instead
        Args:
            :param env: a tuple of environment instance and environmental configuration
            :param model: a tuple of model class and model configuration
            :param stop: dict of running stop condition
            :param running_params: other configuration to customize the rendering
        Returns:
            None
        """

        self.fit(env, model, stop, **running_params)


class _AlgoManager:
    def __init__(self):
        """An algorithm pool class
        """
        for algo_name in POlICY_REGISTRY:
            setattr(_AlgoManager, algo_name, _Algo(algo_name))

    def register_algo(self, algo_name: str, style: str, script: Any):
        """
        Algorithm registration
        Args:
            :param algo_name: algorithm name
            :param style: algorithm learning style from ["il", "vd", "cc"]
            :param script: a running script to start training
        Returns:
            None
        """
        setattr(_AlgoManager, algo_name, _Algo(algo_name + "_" + style))
        POlICY_REGISTRY[algo_name] = script


algos = _AlgoManager()
