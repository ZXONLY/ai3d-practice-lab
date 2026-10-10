# 发布来源与完整历史

本仓库根目录包含可直接安装和运行的二维 Flow Matching 教学代码。

原始教学代码提交：`8bcae187347082e757f9b977d35fbe45c5fd68b1`。原始历史共 3 次提交，完整保存在仓库根目录的 `ai3d-practice-lab.bundle`。在线仓库通过 GitHub 接口发布，发布提交号与原始提交号不同；除本文、README 的发布说明和 transfer 文档的在线获取说明外，原始教学文件未改动。

Git bundle SHA-256：`3efff6eb0076313459f490c56bc763887892c6a759032fc6384daaea0bdbf93a`

恢复原始历史：

```bash
git clone ai3d-practice-lab.bundle ai3d-practice-lab-original
git -C ai3d-practice-lab-original log --oneline
git -C ai3d-practice-lab-original rev-parse HEAD
```

最后一条命令应输出上述原始提交号。普通 `git clone` 在线仓库即可运行根目录教学代码，无需先恢复 bundle。仓库没有打包私有凭证、真实个人数据或本地运行目录。
