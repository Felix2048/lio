# LIO / Escape Room 修复汇总

## 已完成

| 问题 | 修改 | 版本 |
|---|---|---|
| planner 观察漏传实际动作和原始奖励 | `room_symmetric_tax.py` 传递真实动作/奖励，数组空值判断使用 `is not None` | `e4efc75` |
| conda 激活后 CUDA DLL 搜索路径丢失 | `lio/utils/runtime.py` 在 TF 导入前探测当前环境、base 和 CUDA_PATH；仅设置当前进程路径 | `e4efc75` |
| 原入口路径和参数不完整 | 现有 `train_tax.py` 增加显式 CLI、导入路径、终端/文件日志、TensorBoard、运行状态 | `e4efc75` |
| 缺少实际训练更新检查 | worker/planner 增加可选有限 loss/梯度、参数变化与 CUDA 计算验证 | `e4efc75` |
| N=3 忽略 `env.randomize=false`，训练随机出生，奖励可达 9/10 | tax 环境读取配置，缺省固定起点；评估仍固定起点 | 本次 |
| PG 基线同样按人数硬编码出生位置 | 基线增加 randomize 参数，train_pg 传入配置 | 本次 |

原始 worker 奖励之和仍用于 planner 目标，普通 worker 使用 shaping 后奖励。没有把两者混为同一指标。

## 本次验证

- 先复现旧代码的初始化失败，再检查修复通过。
- 6 项回归测试覆盖 N=2/N=3、固定/随机初始化、评估结束后返回训练、缺省配置，以及穷举 5 步内动作路径的总奖励上限。
- 固定起点时原始回合总奖励上限：N=2 为 9，N=3 为 8。
- 两组各 6 个 CPU 训练 episodes，数值检查开启，完成 3 次评估并保存 checkpoint；训练事件未超过奖励上限。
- 从本仓库根目录执行：`conda run -n lio python -I -B tests/test_escape_room_initialization.py`。
- 旧结果保留原样；修复改变初始状态分布，应使用新目录重新训练。

## 尚未修复

**Escape Room planner 的补贴分配梯度问题仍在。** `tax_agent.py` 的 `log_pi_a_s` 计算 `log(sum(tax, subsidy, budget_ratio))`，并非实际采样动作的 log probability。softmax 补贴比例之和恒为 1，补贴输出头在该目标中的直接梯度被抵消。

真实 TaxAgent 的 CPU 图检查中，补贴 kernel/bias 梯度均为 0；分别把补贴全部给 agent 0、agent 2，loss 都为 -96.90579223632812，差值为 0。共享特征仍可间接改变补贴输出，但缺少针对补贴对象的有效优化信号。需连同动作分布、动作缓存和训练目标一起核对，本次没有修改此算法。

社会福利收敛与奖励重分配机制正确是独立检查项，不能将该问题列入已完成修复。

## 同步说明

同步 README、运行依赖和输出忽略规则，保留许可证。保留本仓库的完整实验与历史，不同步精简目录中对 IPD、LISCA 等文件的删除。
