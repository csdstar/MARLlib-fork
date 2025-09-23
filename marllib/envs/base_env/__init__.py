# 环境注册表：用于存储和管理所有支持的多智能体环境
# 键为环境名称字符串（如"mpe"），值为对应的环境封装类或错误信息
ENV_REGISTRY = {}

# 尝试导入并注册"gymnasium_mamujoco"环境
# RLlibGymnasiumRoboticsMAMujoco是该环境的封装类，适配RLlib框架
try:
    from marllib.envs.base_env.gymnasium_mamujoco import RLlibGymnasiumRoboticsMAMujoco
    ENV_REGISTRY["gymnasium_mamujoco"] = RLlibGymnasiumRoboticsMAMujoco
# 若导入失败（如依赖未安装），存储错误信息而非崩溃
except Exception as e:
    ENV_REGISTRY["gymnasium_mamujoco"] = str(e)


# 注册"MPE"环境（Multi-Agent Particle Environment，多智能体粒子环境）
# 这是MARL研究中常用的协作/竞争场景环境（如推箱子、通信任务）
try:
    from marllib.envs.base_env.mpe import RLlibMPE
    ENV_REGISTRY["mpe"] = RLlibMPE
except Exception as e:
    ENV_REGISTRY["mpe"] = str(e)


# 注册基于Gymnasium封装的MPE环境
try:
    from marllib.envs.base_env.gymnasium_mpe import RLlibMPE_Gymnasium
    ENV_REGISTRY["gymnasium_mpe"] = RLlibMPE_Gymnasium
except Exception as e:
    ENV_REGISTRY["gymnasium_mpe"] = str(e)


# 注册"MAMujoco"环境（多智能体 Mujoco 物理仿真环境）
# 常用于需要物理动力学模拟的多智能体任务（如多机械臂协作）
try:
    from marllib.envs.base_env.mamujoco import RLlibMAMujoco
    ENV_REGISTRY["mamujoco"] = RLlibMAMujoco
except Exception as e:
    ENV_REGISTRY["mamujoco"] = str(e)


# 注册"SMAC"环境（StarCraft Multi-Agent Challenge，星际争霸多智能体挑战）
# 基于星际争霸2的战术协作环境，是MARL算法评估的标准环境之一
try:
    from marllib.envs.base_env.smac import RLlibSMAC
    ENV_REGISTRY["smac"] = RLlibSMAC
except Exception as e:
    ENV_REGISTRY["smac"] = str(e)


# 注册"football"环境（Google Football，谷歌足球）
# 多智能体足球协作环境，支持11v11等多种比赛模式
try:
    from marllib.envs.base_env.football import RLlibGFootball
    ENV_REGISTRY["football"] = RLlibGFootball
except Exception as e:
    ENV_REGISTRY["football"] = str(e)


# 注册"magent"环境（多智能体群体环境）
# 包含多种群体协作/竞争场景（如捕食者-猎物、领地争夺）
try:
    from marllib.envs.base_env.magent import RLlibMAgent
    ENV_REGISTRY["magent"] = RLlibMAgent
except Exception as e:
    ENV_REGISTRY["magent"] = str(e)


# 注册"rware"环境（Robot Warehouse，机器人仓库）
# 多机器人在仓库中协作搬运货物的环境
try:
    from marllib.envs.base_env.rware import RLlibRWARE
    ENV_REGISTRY["rware"] = RLlibRWARE
except Exception as e:
    ENV_REGISTRY["rware"] = str(e)


# 注册"lbf"环境（Level-Based Foraging，基于等级的觅食）
# 多智能体协作觅食任务，智能体有不同能力等级
try:
    from marllib.envs.base_env.lbf import RLlibLBF
    ENV_REGISTRY["lbf"] = RLlibLBF
except Exception as e:
    ENV_REGISTRY["lbf"] = str(e)


# 注册"pommerman"环境（炸弹人游戏）
# 混合协作与竞争的多智能体环境，目标是放置炸弹消灭对手
try:
    from marllib.envs.base_env.pommerman import RLlibPommerman
    ENV_REGISTRY["pommerman"] = RLlibPommerman
except Exception as e:
    ENV_REGISTRY["pommerman"] = str(e)


# 注册"hanabi"环境（花火游戏）
# 需要团队协作和通信的卡牌游戏环境（智能体有部分不可见信息）
try:
    from marllib.envs.base_env.hanabi import RLlibHanabi
    ENV_REGISTRY["hanabi"] = RLlibHanabi
except Exception as e:
    ENV_REGISTRY["hanabi"] = str(e)


# 注册"metadrive"环境（元驱动）
# 多智能体自动驾驶环境，支持复杂交通场景模拟
try:
    from marllib.envs.base_env.metadrive import RLlibMetaDrive
    ENV_REGISTRY["metadrive"] = RLlibMetaDrive
except Exception as e:
    ENV_REGISTRY["metadrive"] = str(e)


# 注册"mate"环境（多智能体任务执行）
# 多机器人在动态环境中协作完成任务的环境
try:
    from marllib.envs.base_env.mate import RLlibMATE
    ENV_REGISTRY["mate"] = RLlibMATE
except Exception as e:
    ENV_REGISTRY["mate"] = str(e)


# 注册"gobigger"环境（多多自走棋）
# 多人在线战斗竞技环境，目标是收集资源并击败对手
try:
    from marllib.envs.base_env.gobigger import RLlibGoBigger
    ENV_REGISTRY["gobigger"] = RLlibGoBigger
except Exception as e:
    ENV_REGISTRY["gobigger"] = str(e)


# 注册"overcooked"环境（胡闹厨房）
# 多智能体协作烹饪的环境，需要分工合作完成订单
try:
    from marllib.envs.base_env.overcooked import RLlibOverCooked
    ENV_REGISTRY["overcooked"] = RLlibOverCooked
except Exception as e:
    ENV_REGISTRY["overcooked"] = str(e)


# 注册"voltage"环境（电压控制）
# 电力系统中多智能体协同控制电压的工业环境
try:
    from marllib.envs.base_env.voltage import RLlibVoltageControl
    ENV_REGISTRY["voltage"] = RLlibVoltageControl
except Exception as e:
    ENV_REGISTRY["voltage"] = str(e)


# 注册"aircombat"环境（近距离空战）
# 多智能体飞行器空战环境
try:
    from marllib.envs.base_env.aircombat import RLlibCloseAirCombatEnv
    ENV_REGISTRY["aircombat"] = RLlibCloseAirCombatEnv
except Exception as e:
    ENV_REGISTRY["aircombat"] = str(e)


# 注册"hns"环境（Hide and Seek，捉迷藏）
# 多智能体追逃环境，包含隐藏、搜索、合作等元素
try:
    from marllib.envs.base_env.hns import RLlibHideAndSeek
    ENV_REGISTRY["hns"] = RLlibHideAndSeek
except Exception as e:
    ENV_REGISTRY["hns"] = str(e)


# 注册"sisl"环境（斯坦福智能系统实验室的多智能体环境）
# 包含自动驾驶、机器人协作等多种场景
try:
    from marllib.envs.base_env.sisl import RLlibSISL
    ENV_REGISTRY["sisl"] = RLlibSISL
except Exception as e:
    ENV_REGISTRY["sisl"] = str(e)
