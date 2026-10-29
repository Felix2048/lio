# Escape Room

基于 [Learning to Incentivize Others](https://github.com/011235813/lio)，加入 LOPT 的 Escape Room 训练代码。

以下命令在 PowerShell 中从 `LOPT` 根目录执行。

## 环境

```powershell
conda create -n lio python=3.9.21 pip=23.0.1 setuptools=65.6.3 wheel=0.38.4
conda activate lio
python -m pip install --no-user -r ./lio/requirements.txt
```

GPU 使用 TensorFlow 2.10.1、CUDA 11.2 和 cuDNN 8.1。

## 运行

```powershell
conda activate lio
New-Item -ItemType Directory -Force -Path ./logs | Out-Null
$env:CUDA_VISIBLE_DEVICES = '0'
```

### N=2，M=1

```powershell
python -u -I -B .\lio\lio\alg\train_tax.py er `
    --n_agents 2 `
    --exp_name escape_room `
    --device cuda `
    --episodes 50000 `
    --eval_period 100 `
    --n_eval 10 `
    --seed 12340 `
    --lr_actor 0.001 `
    --loss_penalty_term_weight 0.95 `
    --gamma 0.99 `
    --epsilon_start 0.5 `
    --epsilon_end 0.05 `
    --epsilon_div 100 `
    --entropy_coeff 0.01 `
    --use_tax `
    --use_bank `
    --return_agent_actions `
    --return_agent_rewards `
    --summarize `
    --verify_gpu `
    2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath .\logs\escape-room-n2-console.log
```

### N=3，M=2

```powershell
python -u -I -B .\lio\lio\alg\train_tax.py er `
    --n_agents 3 `
    --exp_name escape_room `
    --device cuda `
    --episodes 50000 `
    --eval_period 100 `
    --n_eval 10 `
    --seed 12340 `
    --lr_actor 0.001 `
    --loss_penalty_term_weight 0.95 `
    --gamma 0.99 `
    --epsilon_start 0.5 `
    --epsilon_end 0.05 `
    --epsilon_div 100 `
    --entropy_coeff 0.01 `
    --use_tax `
    --use_bank `
    --return_agent_actions `
    --return_agent_rewards `
    --summarize `
    --verify_gpu `
    2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath .\logs\escape-room-n3-console.log
```

通过 `--seed` 设置随机种子，`--exp_name` 设置实验名。终端日志保存在 `logs/`，训练结果保存在 `lio/lio/results/`。

许可证见 [LICENSE](LICENSE)。
