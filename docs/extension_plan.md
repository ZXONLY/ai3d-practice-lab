# 扩展路线：从二维速度场到 ShapeVAE / Hunyuan

本文件是后续实施计划，**不是已实现功能列表**。当前可以运行的是二维 Flow Matching；没有 ShapeVAE 模型类、Hunyuan adapter、权重下载器或图像转 mesh 命令。也没有用随机数组冒充真实 3D latent 的接口。

这里以 Hunyuan3D 2.0 的公开结构作为学习参照，不把其他版本的权重、API 和训练协议混用。它将形状生成与纹理生成拆开；形状生成使用 flow-based diffusion transformer，纹理另有生成模型。[Hunyuan3D 2.0 论文](https://arxiv.org/abs/2501.12202)

## 一张机制对照表

| 已运行的二维实验 | 以后真正引入的 3D 环节 | 能沿用与必须重做的部分 |
| --- | --- | --- |
| 目标 `z1` 是二维点 | 目标是几何编码后的 latent | FM 机制可沿用；shape、尺度、分布完全需要重新确认 |
| 高斯噪声 `[B,2]` | 与模型 latent shape 匹配的噪声 | 不可把二维张量简单改名为 mesh |
| MLP 输入坐标与 t | latent token、t，以及条件编码 | 需新的网络和明确 conditioning 接口 |
| Euler / midpoint 更新二维状态 | scheduler 更新 latent 状态 | 必须先对齐时间方向、输出参数化和 scheduler |
| 点云直接可画 | decoder 产生几何表示，再提取 mesh | 重建／解码是新增任务，不等同于 FM |
| 二维分布距离与散点图 | 几何、拓扑、法线、条件一致性与资源评估 | 二维指标不能直接作为 3D 质量证明 |

进入每一阶段前先满足上一阶段的通过标准，不需要一开始就下载整个大模型。

## 阶段 0：把现有二维实验变成可信基线

**输入：**仓库配置和合成数据。  
**运算：**独立配对、构造路径、训练速度场、积分采样。  
**输出：**checkpoint、loss、点云、轨迹、速度场、同 NFE 比较记录。

通过标准：

- 完成实验记录单；能指出训练更新参数、推理更新状态的具体代码
- 解释条件速度目标和边缘速度场的区别
- 已保存代码、配置、seed、环境、指标和图片，能重跑
- 把数值积分误差、速度回归误差和有限采样误差分开讨论

**此阶段不新增 3D 依赖。**

## 阶段 1：只做 ShapeVAE 重建，不做生成模型

### 要回答的问题

给定一个已经存在的简单 mesh，编码再解码，信息在哪里丢失？这与“从噪声生成一个新形状”有什么区别？

### 先做来源和接口核对

1. 锁定上游仓库 commit、模型版本、配置与权重 revision，检查对应许可和使用范围
2. 确认该版本的 checkpoint **包含 encoder**；仅有用于生成的 decoder 权重不足以完成 mesh→latent 编码实验
3. 检查输入 mesh 的坐标归一化、表面采样、法线／其他特征约定
4. 看清 `encode` 的实际返回类型：tensor、后验分布还是其他结构；是否取后验均值或随机采样，是否有 latent scaling
5. 记录 latent 真实 shape、dtype、均值、标准差和峰值内存，禁止从二维例子猜一个 shape

上游提供 [minimal VAE demo](https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/minimal_vae_demo.py)，其中明确展示带 encoder 的 checkpoint、表面采样、encode、decode 和 mesh 导出的分步流程。它是待核对的参考代码；本仓库没有替你执行或下载其中权重。

### 最小可交付实验

**输入：**有权使用的一个简单 mesh；记录其单位、坐标范围和预处理。  
**运算：**surface sampling → encoder → latent → decoder → surface extraction。  
**输出：**原 mesh、重建 mesh、预处理参数、latent 统计、可重跑命令和资源日志。

检查内容：

- 编码和解码均能独立成功，不借助形状生成 DiT
- 同一相机和尺度下比较原件／重建；检查缺失结构、孔洞、浮片、法线和细节
- 对齐坐标和采样协议后再计算几何误差，例如采样表面的 Chamfer distance；注明定义、单位和采样数
- 分开记录 encoder、decoder 与 mesh 提取的耗时和峰值显存
- 记录重建局限；不要把 decoder 造成的细节丢失归咎于尚未引入的 Flow Matching

**停止条件：**若权重不含 encoder、许可未核实、资源不够或上游依赖与 V100 不兼容，先明确阻塞，不用假 latent 或空函数假装重建成功。

## 阶段 2：在真实、冻结的 ShapeVAE latent 上学小型 FM

### 要回答的问题

二维“搬运分布”的训练循环，放到已编码的形状 latent 后，哪些部分仍成立？

前提是阶段 1 可靠，且有一小批来源明确、许可允许的形状数据。缓存 latent 时保存数据 ID、预处理、encoder 权重版本、latent scaling 和 shape 信息。训练／验证要按原始形状分开，避免同一形状的不同表面采样同时进入两边。

**输入：**冻结 encoder 得到的真实 latent。  
**运算：**匹配 shape 的噪声 → 明确路径与目标速度 → 小型可训练 latent 速度模型。  
**输出：**latent FM checkpoint；固定噪声积分后的 latent；经冻结 decoder 解码的 mesh。

设计时必须明确：

- latent 是向量还是 token 集合，是否具有顺序／空间结构
- 标准化的统计量只来自训练集，推理时按相同协议还原
- 网络输出 shape 与监督速度完全一致
- 选择直线路径是新实验的设计，不能声称它逐项复现上游 scheduler
- ShapeVAE 是否冻结，哪些参数进入优化器；明确打印可训练参数量
- 第一版先做无条件生成，图像 conditioning 作为后续单独变量

通过标准：

- 编解码重建基线仍然通过；生成故障不破坏该基线
- loss 与生成 mesh 均有记录；能区分 VAE 的重建上限和 FM 的生成误差
- 使用同 checkpoint／同初始噪声进行 solver 对比，并记录 NFE
- 小数据是否过拟合、是否只产生近似重复形状，有独立检查

这是一个新的轻量研究实验，**不是训练出 Hunyuan3D，也不保证 V100 上可完成任意规模训练**。

## 阶段 3：读懂并验证官方 Hunyuan 形状推理链路

这一阶段优先使用官方已训练模型做最小推理，并跟踪 tensor 流动；不从零训练生产级 DiT。

上游 [形状示例](https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/examples/shape_gen.py) 展示图像进入形状 pipeline 并导出 mesh；[pipeline 源码](https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/hy3dgen/shapegen/pipelines.py) 是追踪 conditioner、model、scheduler 与 VAE 的实际入口。运行前再次核实锁定版本，而不是跟着可变的 main 分支做不可复现安装。

建议按输入→运算→输出记录：

1. 图像预处理：输入模式、分辨率、背景／mask 约定
2. 条件编码：condition tensor 的 shape、dtype 和含义
3. latent 初始化：噪声 shape、seed、尺度
4. 时间与 scheduler：时间方向、离散网格、模型到底预测 velocity / noise / 其他量
5. transformer 前向：输入 latent 与条件怎样交互，输出怎样进入 scheduler
6. latent 解码：缩放、decoder、几何查询、surface extraction
7. mesh 导出：格式、坐标系、可打开性、顶点／面及异常检查

通过标准：锁定版本的官方最小示例真实运行成功，生成 mesh 可以打开；输出、日志、峰值显存和时间全部留存。若 V100 上卡在 bf16、attention kernel、自定义编译或显存，不应宣称“二维 PyTorch 支持，所以 Hunyuan 也支持”。需要独立诊断，再决定可行的受支持路径。

## 阶段 4：最后才讨论微调和纹理

在形状链路稳定之前不引入纹理。形状与纹理的训练目标、显存和依赖不同；上游也有 [先形状再纹理的示例](https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/examples/textured_shape_gen.py)。

真正开始微调前另行确定：

- 具体任务和验收集：要改善哪类形状或条件一致性？
- 数据及模型许可、预算和存储安排
- 训练代码是否覆盖目标版本，哪些参数更新、哪些冻结
- 资源估算包含参数、梯度、优化器、activation、数据缓存和解码，不只看权重文件大小
- 评价如何避免只看挑选出的好案例；回归集怎样保持不变

本仓库不提供未经运行的微调命令，也不承诺单张 V100 足够完成这些工作。

## 扩展时的接口原则

可以沿用“数据／路径／模型／loss／solver／评价”的分层思想，但不要把原来的二维专用类无声改成多用途黑盒。

- 先写明确的 shape、dtype、单位、时间方向与返回值协议
- 新阶段使用独立配置、输出目录和可选依赖，保持原 CPU smoke 可运行
- 先做小输入、真实端到端测试，再添加规模和加速
- 每个新增命令必须对应一个真实可运行能力；未实现能力只写在路线里
- 不在普通 `import`、安装或 preflight 时触发大权重下载
- 不把配置、凭证、远程服务器地址或私人数据写入公开仓库

**每次只增加一个不确定性：先学会搬二维点，再理解几何编码，最后把两条链真正接起来。**
