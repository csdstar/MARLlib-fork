import numpy as np

# 导入基础空战环境类（单/多机空战环境）
from marllib.patch.aircombat.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
import torch
import gym
# 导入Gym的空间类，用于定义观测和动作空间
from gym.spaces import Dict as GymDict, Box
# 导入RLlib的多智能体环境基类
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.utils import merge_dicts

# 策略映射字典：定义不同场景下的智能体-策略映射关系
# MARLlib需要此配置来确定如何为不同智能体分配策略
policy_mapping_dict = {
    # 2v2无武器自对战场景：AI vs AI
    "MultipleCombat_2v2/NoWeapon/Selfplay": {
        "description": "aircombat AI vs AI",
        "team_prefix": ("teamA_", "teamB_"),  # 智能体名称前缀（分两队）
        "all_agents_one_policy": False,       # 不使用统一策略
        "one_agent_one_policy": True,         # 每个智能体一个独立策略
    },
    # 2v2无武器对抗基线场景：AI vs Bot
    "MultipleCombat_2v2/NoWeapon/vsBaseline": {
        "description": "aircombat AI vs Bot",
        "team_prefix": ("agent_",),           # 智能体名称前缀（仅AI方）
        "all_agents_one_policy": True,        # 所有AI智能体使用同一策略
        "one_agent_one_policy": True,
    },
    # 4v4无武器对抗基线场景：AI vs Bot
    "MultipleCombat_4v4/NoWeapon/vsBaseline": {
        "description": "aircombat AI vs Bot",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
}


# 空战多智能体环境类：实现MARLlib要求的接口
# 继承自RLlib的MultiAgentEnv，这是MARLlib兼容环境的基础要求
class RLlibCloseAirCombatEnv(MultiAgentEnv):

    def __init__(self, env_config):
        """初始化环境

        Args:
            env_config: 环境配置参数（包含场景名称、最大步数等）
        """
        self.env_args = env_config  # 保存环境配置
        # 根据配置获取具体的空战环境（如多机空战环境）
        self.env = self.get_env(env_config)
        self.num_agents = self.env.num_agents  # 智能体数量

        # 定义观测空间：符合MARLlib要求的字典格式
        # 外层是GymDict，内层"obs"对应实际观测数据
        self.observation_space = GymDict({"obs": self.env.observation_space})
        # 定义动作空间：直接使用基础环境的动作空间
        self.action_space = self.env.action_space
        # 定义 episode 最大步数限制（MARLlib需要此参数控制训练流程）
        self.episode_limit = self.env.config.max_steps

        # 根据场景类型初始化智能体ID列表（MARLlib要求明确的智能体标识）
        if "vsBaseline" in env_config["map_name"]:  # AI对抗基线Bot场景
            self.agents = ["agent_{}".format(i) for i in range(self.num_agents)]
        else:  # AI vs AI自对战场景（分两队）
            agent_dict = self.env.agents
            # 统计A队智能体数量并生成ID
            self.teamA_agent_num = sum(1 if "A" in agent_name else 0 for agent_name in agent_dict.keys())
            self.agents_teamA = ["teamA_{}".format(i) for i in range(self.teamA_agent_num)]
            # 统计B队智能体数量并生成ID
            self.teamB_agent_num = sum(1 if "B" in agent_name else 0 for agent_name in agent_dict.keys())
            self.agents_teamB = ["teamB_{}".format(i) for i in range(self.teamB_agent_num)]
            # 合并所有智能体ID
            self.agents = self.agents_teamA + self.agents_teamB

    def reset(self):
        """重置环境并返回初始观测

        Returns:
            字典类型的观测：键为智能体ID，值为包含"obs"的字典（符合MARLlib格式）
        """
        original_obs, _ = self.env.reset()  # 调用基础环境的重置方法
        obs = {}  # 存储转换后的观测

        # 根据场景类型处理观测格式
        if "vsBaseline" in self.env_args["map_name"]:  # AI vs Bot场景
            for index, agent in enumerate(self.agents):
                obs[agent] = {
                    "obs": np.float32(original_obs[index])  # 转换为float32类型（符合训练要求）
                }
        else:  # AI vs AI场景（分两队处理）
            for index, agent in enumerate(self.agents_teamA):
                obs[agent] = {
                    "obs": np.float32(original_obs[index])
                }
            for index, agent in enumerate(self.agents_teamB):
                obs[agent] = {
                    "obs": np.float32(original_obs[index + self.teamA_agent_num])
                }
        return obs  # 返回符合MARLlib要求的观测字典

    def step(self, action_dict):
        """执行一步环境交互

        Args:
            action_dict: 字典类型的动作：键为智能体ID，值为动作

        Returns:
            obs: 观测字典（同reset格式）
            rewards: 奖励字典（键为智能体ID，值为对应奖励）
            done: 完成状态字典（包含"__all__"键表示全局是否结束）
            info: 额外信息字典（此处为空）
        """
        # 将动作字典转换为列表（基础环境需要按顺序接收动作）
        actions = []
        for key, value in sorted(action_dict.items()):  # 按智能体ID排序确保顺序正确
            actions.append(value)

        # 调用基础环境的step方法，获取交互结果
        o, _, r, d, i = self.env.step(np.array(actions))

        # 处理奖励和观测
        rewards = {}
        obs = {}
        if "vsBaseline" in self.env_args["map_name"]:  # AI vs Bot场景
            for index, agent in enumerate(self.agents):
                rewards[agent] = r[index][0]  # 提取对应智能体的奖励
                obs[agent] = {
                    "obs": np.float32(o[index])  # 处理观测
                }
        else:  # AI vs AI场景（分两队处理）
            for index, agent in enumerate(self.agents_teamA):
                rewards[agent] = r[index][0]
                obs[agent] = {
                    "obs": np.float32(o[index])
                }
            for index, agent in enumerate(self.agents_teamB):
                rewards[agent] = r[index + self.teamA_agent_num][0]
                obs[agent] = {
                    "obs": np.float32(o[index + self.teamA_agent_num])
                }

        # 构造完成状态字典："__all__"表示是否所有智能体都结束（MARLlib强制要求）
        done = {"__all__": True if d.sum() == self.num_agents else False}
        return obs, rewards, done, {}

    def close(self):
        """关闭环境，释放资源（非MARLlib接口要求）"""
        self.env.close()

    def get_env(self, env_args):
        """根据配置获取具体的基础环境

        Args:
            env_args: 环境配置参数

        Returns:
            初始化后的基础环境实例
        """
        # 从场景名称中解析任务类型和场景参数
        task = env_args["map_name"].split("_")[0]
        scenario = env_args["map_name"].split("_")[1]

        # MARLlib专注于多智能体场景，因此不支持单智能体环境
        if task in ["SingleCombat", "SingleControl"]:
            raise ValueError("Can not support the " +
                             task + "environment." +
                             "\nMARLlib is built for multi-agent settings")
        elif task == "MultipleCombat":  # 多机空战场景（在拓展包中实现）
            env = MultipleCombatEnv(scenario)
        else:
            raise NotImplementedError("Can not support the " +
                                      task + "environment.")
        return env

    def get_env_info(self):
        """获取环境元信息（MARLlib需要此信息进行训练配置）

        Returns:
            包含环境关键信息的字典
        """
        env_info = {
            "space_obs": self.observation_space,  # 观测空间
            "space_act": self.action_space,        # 动作空间
            "num_agents": self.num_agents,         # 智能体数量
            "episode_limit": self.episode_limit,   # 最大步数限制
            "policy_mapping_info": policy_mapping_dict  # 策略映射配置
        }
        return env_info