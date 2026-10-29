# LIO / Escape Room 修复汇总

## 已完成

| 问题 | 修改 | 版本 |
|---|---|---|
| planner 观察漏传实际动作和原始奖励 | `room_symmetric_tax.py` 传递真实动作/奖励，数组空值判断使用 `is not None` | `e4efc75` |
| conda 激活后 CUDA DLL 搜索路径丢失 | `lio/utils/runtime.py` 在 TF 导入前探测当前环境、base 和 CUDA_PATH；仅设置当前进程路径 | `e4efc75` |
| 原入口路径和参数不完整 | 现有 `train_tax.py` 增加显式 CLI、导入路径、终端/文件日志、TensorBoard、运行状态 | `e4efc75` |
| 缺少实际训练更新检查 | worker/planner 增加可选有限 loss/梯度、参数变化与 CUDA 计算验证 | `e4efc75` |
| N=3 忽略 `env.randomize=false`，训练随机出生，奖励可达 9/10 | tax 环境读取配置，缺省固定起点；评估仍固定起点 | `853cf75` |
| PG 基线同样按人数硬编码出生位置 | 基线增加 randomize 参数，train_pg 传入配置 | `853cf75` |

原始 worker 奖励之和仍用于 planner 目标，普通 worker 使用 shaping 后奖励。没有把两者混为同一指标。

## 本次验证

- 先复现旧代码的初始化失败，再检查修复通过。
- 6 项回归测试覆盖 N=2/N=3、固定/随机初始化、评估结束后返回训练、缺省配置，以及穷举 5 步内动作路径的总奖励上限。
- 固定起点时原始回合总奖励上限：N=2 为 9，N=3 为 8。
- 两组各 6 个 CPU 训练 episodes，数值检查开启，完成 3 次评估并保存 checkpoint；训练事件未超过奖励上限。
- 从本仓库根目录执行：`conda run -n lio python -I -B tests/test_escape_room_initialization.py`。
- 旧结果保留原样；修复改变初始状态分布，应使用新目录重新训练。

## 补贴策略梯度与跨回合反馈修复（2026-10-29）

### 原因与修改

旧 planner 确定性输出税率、softmax 补贴比例和预算比例，以 `log(sum(outputs))` 代替动作 log probability。补贴比例之和恒为 1，补贴输出头的直接梯度为零；rollout 动作也未用于 loss。共享层变化仍能让补贴漂移，不能视为已学习了补贴分配。

仅修复动作概率仍不足以解决信用分配：普通智能体在回合结束后才更新；本回合 planner 只整形奖励，不改变本回合的普通智能体策略、观察或环境转移。因此 planner 本回合的原始福利分布独立于本回合的税率/补贴动作，不能直接评价它们对学习的影响。

本次修改：

- 使用 logistic-normal 连续动作策略。训练采样高斯潜变量，sigmoid 映射税率/预算，N−1 个相对于最后一个 agent 的 log ratio 经 softmax 映射补贴。评估使用确定性网络输出。
- 缓存实际采样的 `_policy_latent`，更新时读取缓存，计算该动作的 log density，包括 sigmoid/simplex 变换的 Jacobian。不在更新时重新采样，也不沿采样结果反向传播。
- 普通智能体仍每回合更新，planner 默认每 10 回合更新；同一窗口内 planner 参数保持不变，动作保持 on-policy，普通智能体参数跨回合保留。结束时刷新不足 10 回合的尾部窗口。
- planner 动作的目标取普通智能体更新之后、下一回合及以后回合的折扣原始集体收益，跨环境回合累计。剔除当前物理回合的原始福利作为与该动作无关的基线；最后一个回合没有窗口内后续反馈，因此社会福利项梯度为零。planner loss 按窗口回合数平均。
- 非正奖励 agent 的税率仍置零，其无效税率项不参与 log density；补贴始终非负且总和为 1。奖励整形不再原地修改缓存税率数组。
- 银行约束使用 `abs(sum(shaping)) * sum(log_probability)` 的 score-function 估计，以正成本方向加入最小化目标，替代 `abs(sum(log_probability * shaping))`。
- 终止时将银行余额平均退还后清零；成本、实际补贴和 shaping 指标都计入该次退款，避免指标与 worker 真正得到的奖励不一致。完整回合预算平衡时，银行成本为零是预期结果。
- 修复无银行分支未赋值 `shaped_reward_sum` 的异常。
- TensorBoard 增加 `train/planner_updates` 和 `train/planner_window_episodes`，CUDA 检查在普通智能体和 planner 各自首次实际更新时执行。

planner 仍以原始集体收益为目标。没有添加公平分配奖励、拉杆者奖励规则或强制补贴下限。

### 新增配置

三份 tax 配置显式记录以下参数，CLI 可用同名选项覆盖：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `planner_action_std` | 1.0 | 转换前高斯潜变量的采样标准差，必须是有限正数 |
| `planner_update_period` | 10 | planner 更新窗口的回合数，必须为不小于 2 的整数 |

这两个默认值是本次修复选择的实现参数，论文没有给出该采样分布及该跨回合窗口设置，不能标作论文原始超参数。原命令不增加这两个选项时使用上述默认值。已有训练进程不会加载磁盘修改，需重新启动并使用新输出目录。

### 验证

环境：Windows，`lio` conda 环境，Python 3.9.21 / TensorFlow 2.10.1 / NumPy 1.23.0。

- 修改前在原仓库运行补贴梯度回归测试，重现 subsidy kernel 梯度范数为 0 的失败。
- 修复后的受控固定输入：subsidy kernel/bias 梯度范数分别为 25.3787078857 / 6.2992057800；保持缓存动作不变，把第一个补贴 bias 增加 0.1，loss 从 −48.1360092163 变为 −48.5760116577。
- 14 项回归覆盖直接梯度、缓存动作、有限差分、变换密度与税率掩码、采样/评估、真实 buffer 更新、约束符号与轨迹聚合、银行退款及指标、checkpoint、无银行分支及参数校验、跨回合目标、窗口内冻结和尾部刷新。
- 受控两回合学习任务使用真实 `PolicyGradient.train`：第一个 planner 动作影响 worker 学习奖励，只有下一回合的预期集体收益反馈给 planner。2,000 次受控训练后，目标补贴比例从 0.333333 增至 0.819503，验证通过 worker 学习更新的延迟反馈方向。该测试任务不是 Escape Room 收敛实验，其测试收益未写入生产目标。
- N=2 完成 100 个 CPU 回合、5 次评估及 10 次 planner 更新；N=3 完成 1,000 个 CPU 回合、10 次评估及 100 次 planner 更新。启用有限 loss/梯度/参数检查，完成模型保存、TensorBoard 读取和窗口计数校验；训练原始福利上限分别未超过 9 / 8。
- 最终版本 N=3 的 1,000 回合短试末次评估福利为 1.5，仍有波动；不能把执行检查当作收敛或论文结果复现。中间版本的末次福利 7.1 不属于最终版本结果。
- N=3 另完成 20 个 CUDA 回合、4 次评估和 2 次 planner 更新。首次 planner 更新发生于第 10 回合，执行 20 个 GPU MatMul / 16 个 GPU GradientDescent 算子；普通智能体同样通过真实 GPU 更新检查。

从本仓库根目录运行：

```powershell
conda run -n lio python -I -B tests/test_escape_room_initialization.py
conda run -n lio python -I -B tests/test_tax_planner_policy.py -v
conda run -n lio python -I -B tests/check_tax_planner_training.py --n 2
conda run -n lio python -I -B tests/check_tax_planner_training.py --n 3 --episodes 1000 --eval_period 100 --n_eval 10
conda run -n lio python -I -B tests/check_tax_planner_training.py --n 3 --device cuda --episodes 20 --eval_period 5 --n_eval 2
```

训练检查输出写入系统临时目录，不覆盖既有结果。本次只修改 Escape Room；Cleanup 不添加跨回合反馈。

### 后续检验

信用分配窗口是有限的，窗口末尾动作对窗口外学习效果的影响尚未计入；窗口长度和采样标准差会影响方差与学习效果。需要完整训练和多个 seed 检查福利、拉杆者补贴、税后收益及稳定性。暂不将机制已复现作为修复结论。

## 同步说明

同步 README、运行依赖和输出忽略规则，保留许可证。保留本仓库的完整实验与历史，不同步精简目录中对 IPD、LISCA 等文件的删除。
