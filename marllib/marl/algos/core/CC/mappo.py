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

from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG as PPO_CONFIG
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy, KLCoeffMixin, ppo_surrogate_loss
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import EntropyCoeffSchedule, \
    LearningRateSchedule
from ray.rllib.utils.typing import TensorType

from marllib.marl.algos.core import setup_torch_mixins
from marllib.marl.algos.utils.centralized_critic import CentralizedValueMixin, centralized_critic_postprocessing


#############
### MAPPO ###
#############

def central_critic_ppo_loss(policy: Policy, model: ModelV2,
                            dist_class: ActionDistribution,
                            train_batch: SampleBatch) -> TensorType:
    """
    构建集中式PPO的损失函数。
    Args:
        policy (Policy): 需要计算损失的策略。
        model (ModelV2): 需要计算损失的模型。
        dist_class (Type[ActionDistribution]): 动作分布的类。
        train_batch (SampleBatch): 训练数据批次。

    Returns:
        Union[TensorType, List[TensorType]]: 返回一个损失张量，或者是多个损失张量的列表。
    """
    # 初始化集中式值函数
    CentralizedValueMixin.__init__(policy)

    # 使用 PPO 的替代损失函数
    func = ppo_surrogate_loss

    # 保存原始值函数
    vf_saved = model.value_function

    # 配置是否将对手的动作包含在集中式价值函数中
    opp_action_in_cc = policy.config["model"]["custom_model_config"]["opp_action_in_cc"]

    # 定义新的值函数，考虑对手的动作
    model.value_function = lambda: policy.model.central_value_function(train_batch["state"], train_batch["opponent_actions"] if opp_action_in_cc else None)

    # 获取计算得到的集中式价值输出
    policy._central_value_out = model.value_function()

    # 计算PPO的损失
    loss = func(policy, model, dist_class, train_batch)

    # 恢复原始值函数
    model.value_function = vf_saved

    return loss


# 基于 PPOTorchPolicy 的自定义策略，定义了 MAPPO 使用的损失函数、后处理函数、以及训练时的优化器配置
MAPPOTorchPolicy = PPOTorchPolicy.with_updates(
    name="MAPPOTorchPolicy",
    get_default_config=lambda: PPO_CONFIG,
    postprocess_fn=centralized_critic_postprocessing,
    loss_fn=central_critic_ppo_loss,
    before_init=setup_torch_mixins,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        CentralizedValueMixin
    ])


# 根据传入的配置选择使用 Torch 框架的 MAPPOTorchPolicy
def get_policy_class_mappo(config_):
    if config_["framework"] == "torch":
        return MAPPOTorchPolicy


MAPPOTrainer = PPOTrainer.with_updates(
    name="MAPPOTrainer",
    default_policy=None,
    get_policy_class=get_policy_class_mappo,
)
