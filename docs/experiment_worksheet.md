# 实验记录单：先预测，再运行，再解释

目标不是复现一张“好看”的散点图，而是建立能够被代码和证据推翻的判断。复制这份文档作为自己的记录；空白处是待填写项，不是已经得到的结果。

## 0. 记录环境和基线

- 日期：
- Python / PyTorch 版本：
- CPU 或 GPU 型号：
- dtype：float32
- 代码版本或压缩包版本：
- 配置文件及其副本路径：
- 完整训练命令：
- checkpoint 路径：
- 训练 seed / 采样 seed：
- 实际训练步数 / batch size / learning rate：
- 是否全部通过单元测试：
- 未运行或失败的检查：

先运行 README 的 preflight、测试和 smoke。若 smoke 失败，保存完整报错，先定位导入、数据 shape、loss、梯度或保存／加载中的哪一步出问题。不要进入调参循环。

## 1. 实验 A：模型不变，采样步数会改变什么？

### A1. 运行前写预测

问题：固定同一个 checkpoint 和同一批初始噪声，Euler 从 4 步增加到 32 步，结果会怎样？

- 我的预测：
- 依据：
- 哪个结果会让我认为预测不成立：
- 我预期变化的是模型权重还是采样状态：
- 我预测验证 MSE 是否因此改变，为什么：

合理的假设是：步数增加可能减小积分误差；但对真实目标分布的误差不保证严格单调下降，因为速度场本身也有误差。不要把这个假设先写成结果。

### A2. 建立一次基线

```bash
python -m flowlab train --config configs/moons.json --out runs/moons --device cpu
python -m flowlab compare --checkpoint runs/moons/checkpoint.pt --out runs/compare --steps 4 8 16 32 64 --seed 123 --samples 2048 --device cpu
```

本实验只训练一次。之后所有比较都加载 `runs/moons/checkpoint.pt`。用 `compare` 的共同噪声、共同参考目标、共同投影方向控制随机性。不是“每种 solver 再训练一个模型”。

如果这个目录已有基线，不要覆盖；改成另一个有意义的目录名并在下面记录。

### A3. 收集证据

从 `runs/compare/comparison.json` 抄录数值，从对应的 `samples_<method>_<steps>.png` 观察图像，并保留原文件路径：

| method | steps | NFE | sliced_wasserstein_2 | 终点点云观察 | 证据路径 |
| --- | ---: | ---: | ---: | --- | --- |
| euler | 4 | 4 | 待测 | | |
| euler | 8 | 8 | 待测 | | |
| euler | 16 | 16 | 待测 | | |
| euler | 32 | 32 | 待测 | | |
| euler | 64 | 64 | 待测 | | |
| midpoint | 4 | 8 | 待测 | | |
| midpoint | 8 | 16 | 待测 | | |
| midpoint | 16 | 32 | 待测 | | |
| midpoint | 32 | 64 | 待测 | | |
| midpoint | 64 | 128 | 待测 | | |

看图时具体描述：两瓣有没有缺失？月牙之间的间隙有没有被填满？尾部是否异常？分布厚度是否接近目标？不要只写“变好”。绘图轴范围也应一致，否则视觉上容易误判。

### A4. 分开回答两个问题

1. **同样离散步数**：Euler 16 对 midpoint 16。哪种结果更接近目标？成本有什么变化？
2. **同样网络调用预算**：Euler 32 对 midpoint 16。判断是否仍成立？

如果只报告问题 1，就不能得出“midpoint 在相同算力下更好”。NFE 是 batch 级网络求值次数，并不是每个点分别计一次，也不是 FLOPs 或秒数。

### A5. 用运算解释，避免事后编故事

- 观察到的事实：
- 与预测一致／不一致的部分：
- 哪一个公式或代码位置能解释它：
- 是否足以区分积分误差、模型误差和有限采样误差：
- 目前不能得出的结论：

可以提出一个后续检验，而不是将所有现象都解释成“模型不够大”。例如在相同 checkpoint 上再用 seed=124 的共同噪声重新 compare，看看方法排序是否稳定。

## 2. 实验 B：只改变训练步数

现在改变训练过程。保持数据、模型、学习率、batch size 和 seed 不变，只将基线训练步数加倍。先预测“更久训练”的收益是否已接近平台期。

### B1. 复制配置并只改一项

```bash
python - <<'PY'
import json
from pathlib import Path
source = Path('configs/moons.json')
config = json.loads(source.read_text())
old = config['steps']
config['steps'] = old * 2
out = Path('configs/moons-longer.json')
out.write_text(json.dumps(config, indent=2) + '\n')
print(f'{source}: steps={old}; {out}: steps={config["steps"]}')
PY
python -m flowlab train --config configs/moons-longer.json --out runs/moons-longer --device cpu
python -m flowlab sample --checkpoint runs/moons/checkpoint.pt --out runs/baseline-m32 --method midpoint --steps 32 --seed 123 --samples 2048 --device cpu
python -m flowlab sample --checkpoint runs/moons-longer/checkpoint.pt --out runs/longer-m32 --method midpoint --steps 32 --seed 123 --samples 2048 --device cpu
```

两次是从头训练的独立运行，不是 checkpoint resume。相同 seed 有助于控制初始化和随机序列，但跨设备、跨软件版本不承诺逐位复现。

### B2. 记录预测和结果

- 运行前预测：
- 唯一改变的变量：训练 steps 从 ___ 到 ___
- 基线最后验证 MSE：
- 更长训练最后验证 MSE：
- loss 曲线平台／振荡／异常：
- 相同 solver、步数、seed 的样本区别：
- 耗时变化及其测量方式：
- 结论只适用于哪些配置和 seed：

如果 loss 更低、图却更差，先核对 checkpoint、seed、相同采样设置和坐标范围，再考虑模型场误差与指标的差异；不能直接认定测试失效。如果 loss 不接近零，回看 README 的条件方差解释。

## 3. 把一批张量讲给别人听

打开 `path.py`、`model.py`、`losses.py`、`train.py` 和 `solvers.py`，写下实际变量名或函数名：

| 项目 | shape / 内容 | 在哪里生成 | 是否由训练更新 |
| --- | --- | --- | --- |
| 起点 `z0` | `[B,2]` 高斯噪声 | | 否 |
| 终点 `z1` | `[B,2]` 目标数据 | | 否 |
| 时间 `t` | 每个样本一个标量 | | 否 |
| 中间状态 `z_t` | `[B,2]` | | 由路径公式构造 |
| 网络输入 | `[B,3]` | | 否 |
| 预测速度 | `[B,2]` | | 随参数变化 |
| 监督速度 | `[B,2]` | | 否 |
| 参数 `θ` | 每层权重与偏置 | | 是，优化器更新 |
| 推理状态 `z_n` | `[B,2]` | | solver 更新，不是优化器 |

补齐这三句话：

1. 训练输出的是 ___，loss 比较的是 ___ 和 ___
2. 推理每一步读取 ___，修改 ___，不会修改 ___
3. 这颗推理噪声没有对应的真实目标点，但整个生成点云应该接近 ___

## 4. 第二轮可以选的单变量实验

只选一项，先写预期，再按同样协议运行：

- 改 `hidden_dim`：区分模型容量不足与训练不足；记录参数量和成本
- 改 `learning_rate`：观察震荡／收敛，不预设学习率越大越好
- 换 `dataset` 为 `gmm8`：这属于新生成任务，必须重新训练；观察是否覆盖八个模态
- 改 `noise`：目标分布也变了，loss 与指标不应直接当成同一个任务的排行榜

每轮都保留一份可重跑的配置、命令和证据。没有运行就写“未运行”，没有证据就写“待检验”。

## 5. 进入 3D 前的通过标准

- 能手算一个 `z_t` 和速度标签，并与实现一致
- 能解释为何条件速度不等于推理时可见的真值
- 能找到训练与采样分别更新哪个张量
- 能完成同 checkpoint / 同初始噪声 / 同 NFE 的对比
- 能用样本质量和验证回归损失回答不同的问题
- 能清楚区分已实现的二维闭环与 [尚未实现的 3D 扩展](extension_plan.md)

达到这些标准后，再引入 ShapeVAE 编解码、latent 尺度和条件输入；不用先把大模型全部下载一遍。
