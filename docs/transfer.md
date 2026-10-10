# 把仓库带到 SSH 服务器

公开发布版本可直接克隆：

```bash
git clone https://github.com/ZXONLY/ai3d-practice-lab.git
cd ai3d-practice-lab
```

原始三次提交完整保存在根目录 `ai3d-practice-lab.bundle` 中，发布提交号与原始提交号不同，详见 [发布来源说明](../PUBLICATION.md)。下面保留离线转移方法；不要把私钥或密码写进本仓库。

## 路线 A：保留提交历史的 Git bundle（推荐）

在 bundle 所在目录执行：

```bash
git clone -b main ai3d-practice-lab.bundle ai3d-practice-lab
cd ai3d-practice-lab
git log --oneline
git status --short
```

这是离线克隆，不要求外部 Git 服务。克隆产生的 `origin` 只是本地 bundle 路径，不是在线托管远端。若需要核验包中的 refs，可执行：

```bash
git bundle list-heads ../ai3d-practice-lab.bundle
git bundle verify ../ai3d-practice-lab.bundle
```

## 路线 B：源码 ZIP

```bash
unzip ai3d-practice-lab-source.zip
cd ai3d-practice-lab
```

源码 ZIP 与 bundle 的最终提交内容一致，但不含 `.git` 历史；想保留历史就选路线 A。不要把 ZIP 里的源码与另一版本的配置混用。

## 运行证据 ZIP

`ai3d-practice-lab-cpu-evidence.zip` 包含本次 CPU 的小型 checkpoint、日志、配置／环境信息、训练图和 solver 对比。它用于检查已有结果，不代替你自己的新实验。checkpoint 只加载本次可信来源或你自己生成的文件；本代码使用 `torch.load(..., weights_only=True)`，但不建议将任何未知权重文件当成安全输入。

源码中的 [CPU 验证报告](../reports/verified_cpu_run/REPORT.md) 给出实际命令、代码提交、指标和尚未验证的边界。证据包中的训练和比较输入 seed 不同，见报告，不能把两个数的变化归因于 solver。

完成解包后，按 [V100 检查清单](v100.md) 建立独立环境。先运行只读预检和 float32 smoke，再做正式实验。当前没有服务器连接信息，也没有任何已完成的 V100 实验。
