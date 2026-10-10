# AI3D Practice Lab：先把 Flow Matching 跑明白

公开仓库获取与原始三次提交历史见 [发布来源说明](PUBLICATION.md)。在线发布提交号与原始教学提交号不同，原始历史完整保存在根目录 Git bundle 中。

一个可读、可改、可验证的二维生成模型实验室。先在 CPU 上把高斯点云搬成双月牙，再把同一条训练／推理主线迁移到 3D latent。

**这版实现的范围：**PyTorch、float32、二维数据、MLP 速度场、独立配对直线路径、MSE 训练，以及 Euler / 显式 midpoint ODE 采样。它不包含 ShapeVAE、Hunyuan3D 权重或图像转 3D；后两者的工作边界见 [扩展路线](docs/extension_plan.md)。

## 先获得一个完整闭环

顺序建议：

1. 在独立环境中安装，运行 preflight 和单元测试
2. 用 smoke 配置确认训练、保存、加载、画图全部连通
3. 用 moons 配置完成一次基线训练
4. 保持 checkpoint 不变，只改采样方法与步数
5. 填写 [实验记录单](docs/experiment_worksheet.md)，用证据解释结果

V100 是后续运行目标，不是本仓库已经验证过的硬件。先读 [V100 检查清单](docs/v100.md)；本仓库不连接 SSH、不修改驱动、不自动下载大模型。

## 1. 安装：使用独立环境

以下为 Linux x86_64 的 CPU 起步命令，建议 Python 3.12。不要在现有 Hunyuan、CUDA 或科研环境里直接升级依赖。

```bash
cd ai3d-practice-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python -m pip check
python -m flowlab preflight --device cpu --out reports/preflight-cpu.json
python -m unittest discover -s tests -v
```

PyTorch 安装命令取自 [官方历史版本页面](https://pytorch.org/get-started/previous-versions/)。这里固定一个可检查的起步版本，不表示它是最新版本；macOS、其他架构或不同 Python 版本请按官方说明选择对应 wheel。V100 使用独立的 GPU 环境，命令见专页。

所有示例都在仓库根目录运行。若已在独立环境装好所需依赖，只是不想 editable install，可使用：

```bash
PYTHONPATH=src python -m flowlab --help
PYTHONPATH=src python -m unittest discover -s tests -v
```

`PYTHONPATH` 只解决源码的导入路径，不会安装 PyTorch，也不会修复 CUDA。遇到没有 torch、无法下载 wheel 或 CUDA 不可用，应先解决对应环境问题；不要把“尚未运行”写成“测试通过”。

## 2. 跑一次，再看产物

### 最小烟雾测试

```bash
python -m flowlab train --config configs/smoke.json --out runs/smoke --device cpu
python -m flowlab sample --checkpoint runs/smoke/checkpoint.pt --out runs/smoke-sample --method midpoint --steps 8 --seed 123 --samples 512 --device cpu
```

smoke 用于发现导入、shape、梯度、checkpoint 或文件输出错误。它不是收敛基准，也不保证漂亮的月牙。

### 正式二维基线

```bash
python -m flowlab train --config configs/moons.json --out runs/moons --device cpu
python -m flowlab sample --checkpoint runs/moons/checkpoint.pt --out runs/sample --method midpoint --steps 32 --seed 123 --samples 2048 --device cpu
python -m flowlab compare --checkpoint runs/moons/checkpoint.pt --out runs/compare --steps 4 8 16 32 64 --seed 123 --samples 2048 --device cpu
```

重复实验时换一个 `--out` 目录，保留旧结果；程序会拒绝写入已有内容的目录。不要只留下效果最好的一张图。

训练输出位于 `runs/moons/`：

| 文件 | 先问它什么问题 |
| --- | --- |
| `checkpoint.pt` | 保存了哪个已训练速度场？之后 sample / compare 用它 |
| `metadata.json` | 这次运行实际用了哪些配置和环境信息？ |
| `metrics.jsonl` | 哪个优化步记录了什么训练／验证值？ |
| `loss.png` | 学习是否有趋势，是否出现非有限值或突变？ |
| `samples.png` | 终点分布覆盖了目标两瓣吗？厚度、间隙和离群点怎样？ |
| `trajectories.png` | 从噪声到终点的积分轨迹是什么样？ |
| `vector_field.png` | 同一空间点在不同时间的速度方向是否变化？ |
| `summary.json` | 本次训练的汇总结果是什么？ |

采样输出还有 `inputs.pt`（初始噪声、参考目标、投影方向）、`generated.pt`（终点、完整轨迹、时间）和 `metadata.json`（含 checkpoint SHA-256）。`compare` 保存相同的 `inputs.pt`、`metadata.json`，以及 `comparison.json`、`comparison.png` 和各组合的 `samples_<method>_<steps>.png`。这些原始数据与来源信息可用于复查，而不必只相信截图。

`checkpoint.pt` 包含模型、优化器状态、配置、step 与部分随机状态。本版 CLI 没有 resume 命令；保存这些状态不等于已实现精确断点续训。仅加载本实验自己生成、来源可信的 checkpoint。

图像是诊断证据。二维点云图不能直接证明 3D 几何质量，损失曲线也不能代替生成样本的检查。

### 可选：逐格运行教学 notebook

`notebooks/01_flow_matching_by_hand.ipynb` 展示张量、单步优化和采样。核心 CLI 与单元测试不依赖 Jupyter；如果想交互运行，在同一个已激活的独立环境中执行：

```bash
python -m pip install -e '.[notebook]'
python -m jupyterlab notebooks/01_flow_matching_by_hand.ipynb
```

选择该虚拟环境的 kernel，按顺序执行。notebook 的代码单元也会被仓库测试按顺序执行，不应只检查 JSON 能打开。

## 3. 训练到底在算什么？

### 三种不同的对象

- **模型参数 `θ`**：MLP 各层权重和偏置；训练时优化器修改它们
- **样本状态 `z_t`**：当前正在搬运的二维点；采样时 ODE solver 修改它
- **时间 `t`**：路径上的位置，范围是 0 到 1；不是训练轮数，也不是实际运行秒数

本实验固定方向：**`t=0` 是高斯噪声，`t=1` 是数据**。读其他仓库时要重新确认方向、符号和 scheduler，不能机械照搬。

### 一批训练数据的输入 → 运算 → 输出

批大小为 B，坐标维数为 2：

1. 训练开始时 `data.py` 生成固定目标训练集；每步随机抽取 B 个 `z1`，shape 为 `[B, 2]`，来自 moons 或 gmm8
2. 独立抽取 `z0 ~ N(0, I)`，shape 为 `[B, 2]`；随机点与目标点没有预先学到的对应关系
3. 抽取 `t ~ Uniform(0, 1)`，shape 为 `[B, 1]`，每个样本一个时间
4. `path.py` 构造 `z_t = (1-t) z0 + t z1`，以及监督速度 `u = z1 - z0`
5. `model.py` 将二维状态和时间拼接为 `[B, 3]`，MLP 输出 `v_θ(z_t, t)`，shape 为 `[B, 2]`
6. `losses.py` 计算预测速度与目标速度的 MSE
7. `train.py` 执行清梯度、反向传播、优化器更新；改变的是 `θ`

这一训练步不需要先从 0 到 1 解一遍 ODE。直线路径让任意中间状态和标签都可以直接计算，这正是这个教学实验的简洁之处。

MSE 若对 B 个样本的两个坐标一起取平均，则为：

```text
L = 1/(2B) × Σ_i Σ_coordinate [vθ(z_t,i, t_i) - (z1,i - z0,i)]²
```

如果另一份代码先对坐标求和、再对 batch 求均值，它的数值可能相差 2 倍，不要直接比较两份 loss 曲线。

### 一个可手算的样本

```text
z0 = ( 1, -1)
z1 = (-1,  2)
t  = 0.25

z_t = 0.75 × (1, -1) + 0.25 × (-1, 2) = (0.5, -0.25)
u   = z1 - z0 = (-2, 3)
MLP 输入 = (0.5, -0.25, 0.25)
MLP 目标 = (-2, 3)
```

`u` 是速度，不是终点坐标。网络也没有拿到这一个样本的 `z1` 作为输入。

### 为什么“每条训练线都很直”，生成轨迹却会弯？

`z1-z0` 是给定这对端点的**条件监督目标**。推理时只有当前位置和时间，网络不知道训练时隐藏的端点。同一 `(z_t,t)` 附近可能出现不同配对、不同方向的监督速度。

平方损失的理想解是条件平均：

```text
v*(z,t) = E[z1-z0 | z_t=z, t]
```

网络学习的是这个**边缘速度场**的近似。它随当前位置和时间改变，沿它积分得到的单条轨迹不必是直线；也不能解释为“恢复这颗噪声原来配对的那个真实点”。本仓库使用独立配对，没有实现 minibatch optimal transport 匹配。条件直线路径不等于已经求解了两个完整分布之间的最优传输。

以上条件场与边缘场的关系对应 [Flow Matching 原论文 §3](https://arxiv.org/html/2210.02747#S3)；独立配对与其他 coupling 的区分也可参考 [generalized conditional flow matching 论文](https://arxiv.org/abs/2302.00482)。

### 为什么验证 MSE 不应被要求降到零？

用平方损失的条件期望性质，可以把期望误差分成两项：

```text
E||vθ - u||²
= E||vθ - E[u | z_t,t]||² + E||u - E[u | z_t,t]||²
    可学习的速度场误差             隐藏端点带来的条件方差
```

第二项通常非零，而且不由模型参数控制。这是本实验的条件回归标签噪声，不必然是实现 bug。有限网络、有限训练和有限验证样本还会带来额外误差。

本实现使用独立生成、整次训练固定的验证三元组，每次记录都评估相同的验证问题；训练 loss 是上一记录区间内的 batch loss 均值。最终 checkpoint 来自最后一个训练 step，不按最低验证损失挑选。

因此：

- 验证 MSE 用于监控速度回归，不是“图片质量分数”
- 更低验证损失不保证任何 seed、任何 solver 步数下样本都更好
- 采样还受数值积分误差影响；只加采样步数也不能补回没学好的速度场
- 用新随机验证样本所得的小幅差异可能只是 Monte Carlo 波动；比较前要确认验证协议

## 4. 推理到底在改什么？

```text
checkpoint 中的 θ + 新的 z(0) ~ N(0,I)
        ↓ 固定 θ，求解 dz/dt = vθ(z,t)
Euler 或 midpoint 逐步更新 z
        ↓ 到 t=1
二维生成点云
```

这里不开优化器、不反向传播、不更新权重。把采样 `--steps` 从 8 改成 32 不会多训练模型；只是更细地积分同一个已训练速度场。推理仍执行神经网络前向运算。

固定步数 N，`h=1/N`：

```text
Euler:
    k1 = vθ(z_n, t_n)
    z_(n+1) = z_n + h × k1
    每步 1 次网络调用

显式 midpoint:
    k1 = vθ(z_n, t_n)
    k2 = vθ(z_n + h/2 × k1, t_n + h/2)
    z_(n+1) = z_n + h × k2
    每步 2 次网络调用
```

midpoint 的中点是临时估计，不是真实目标点。小步长且速度场足够平滑时，高阶方法通常更准确，但实际质量还取决于模型误差和步长，不能保证每张图都更好。

### 公平比较必须固定什么？

`compare` 保持相同 checkpoint、初始噪声、目标参考样本和 128 个投影方向，比较两种 solver 的多个步数。主要记录 `sliced_wasserstein_2` 和网络求值次数 NFE。`seconds_single_run` 是未预热的单次采样计时，不能当作可靠吞吐基准。

- 同步数：Euler 16 的 NFE=16，midpoint 16 的 NFE=32；这是离散化策略对比，不是同计算预算
- 同 NFE：可以在输出中比较 Euler 32 与 midpoint 16
- 相同 NFE 也不严格等于相同墙钟时间；GPU 启动开销、同步、batch 大小和实现都影响计时
- Sliced Wasserstein-2 是沿有限方向投影的分布差异估计，越小通常越接近本次参考点云；它不是完整高维 Wasserstein 距离，不是百分比，也没有跨数据集通用及格线
- 应结合点云、轨迹、多个 seed 与参考样本的采样误差看结论，不要只按一个最小数排序

## 5. 从哪个文件开始改？

| 想理解／改变的部分 | 入口 |
| --- | --- |
| moons / gmm8 数据怎样生成 | `src/flowlab/data.py` |
| 为什么输入是中间状态，标签是速度 | `src/flowlab/path.py` |
| MLP 怎样使用坐标和时间 | `src/flowlab/model.py` |
| 预测与监督怎样变成一个标量 | `src/flowlab/losses.py` |
| 哪一行真正更新模型权重 | `src/flowlab/train.py` |
| 哪一行真正更新采样状态 | `src/flowlab/solvers.py` |
| 怎样保存／恢复模型 | `src/flowlab/checkpoint.py` |
| 如何定量比较分布 | `src/flowlab/evaluation.py` |
| 点云、轨迹和速度场怎么画 | `src/flowlab/plotting.py` |
| CLI 参数如何进入运算 | `src/flowlab/cli.py` |

`configs/moons.json` 的起步设置为：2,000 个优化步，batch size 256，Adam 学习率 0.001，3 个宽度为 128 的隐藏层，8,192 个训练目标点，2,048 个固定验证样本，每 100 步记录一次，数据噪声标准差 0.08，CPU threads=1，seed=42。`gmm8.json` 提供另一种目标分布；全部数据由本地生成，不下载数据集。

第一次只改一个变量。优先改采样步数，再改训练 `steps` 或网络 `hidden_dim`；不要同时改数据、学习率、模型宽度和 solver 后猜测原因。配置中的训练 `steps` 与 CLI 中的采样 `--steps` 是两个不同概念。

## 6. 怎样算这次真的学会了？

你应能不用术语堆叠，回答以下问题：

1. 训练时输入的三个数是什么，输出的两个数是什么？
2. MSE 对齐的两个张量各来自哪一行代码？
3. 训练更新 `θ`；采样更新 `z`。能分别指出代码位置吗？
4. 推理为什么不需要从数据集里找一个 `z1`？
5. 条件速度标签为什么会冲突，loss 为什么通常不为零？
6. 为什么 Euler 32 与 midpoint 32 的计算量不同？
7. 换一个目标分布时，需要重新训练哪部分；为什么不能只换画图标签？
8. 迁移到 ShapeVAE latent 后，哪些机制不变，哪些 shape、尺度、条件和解码环节必须重做？

继续：[第一份实验记录](docs/experiment_worksheet.md) · [V100 环境预检](docs/v100.md) · [ShapeVAE / Hunyuan 路线](docs/extension_plan.md)

## 7. 交付与复现

- [离线 Git bundle / 源码 ZIP 使用方法](docs/transfer.md)
- [本次 CPU 实测报告](reports/verified_cpu_run/REPORT.md)
- `requirements-tested-cpu.txt` 记录本次 CPU 顶层依赖版本；不是完整传递依赖锁文件
