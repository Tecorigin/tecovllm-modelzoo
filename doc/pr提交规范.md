# 提交 PR

在完成模型开发及贡献模型所需的全部文件后，您可以通过 PR（Pull Request）将代码提交到本仓库。

本文档介绍提交 PR 的规范要求与操作方式。请在提交前按规范逐项检查，所有内容符合要求后再发起 PR。全文包含以下两部分：

| 章节 | 内容 |
|------|------|
| [1. PR 提交规范](#1-pr-提交规范) | 路径命名、功能验证、注释与 License 声明、代码风格 |
| [2. 提交 PR](#2-提交-pr) | 如何发起 PR 以及 PR 标题和说明的填写要求 |

---

## 1. PR 提交规范

提交 PR 前，请从以下四个方面进行检查：

- [ ] 路径与命名规范
- [ ] 功能实现验证
- [ ] 注释与 License 声明
- [ ] （可选）代码优化

### 1.1 检查功能实现

提交的内容必须通过以下两项验证：

| 验证项 | 要求 |
|--------|------|
| **推理服务验证** | 按适配指南替换模型文件后，TecovLLM 服务可正常启动，单样本推理请求返回正常结果 |
| **精度验证** | 在太初提供的 Docker 环境中，使用 TecovLLM 服务的 OpenAI 接口对 Evalscope 标准测试数据集完成推理，精度符合要求 |

### 1.2 检查注释及 License 声明

#### 注释要求

对代码中的关键部分添加注释，帮助使用者快速理解代码结构，包括但不限于：

- 函数的功能说明（如前后处理相关的 `resize`、`nms` 等）
- IO 相关代码（`init`、`save`、`load` 等）
- 运行过程中的关键状态（如 `print` 日志、保存模型等）

#### License 声明

所有完全自主开发的代码文件，必须在文件最上方添加版权声明和开源许可声明。Tecorign ModelZoo 提供了两种语言的 License 模板，请根据代码语言选择：

**C/C++ License**
```c
// BSD 3-Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
// reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
```

**Python License**
```python
# BSD 3-Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
```

> **💡 已有第三方版权声明的文件**：如果原代码文件已有第三方版权声明，为减轻工作量，可以在原文件基础上直接追加一行声明即可，无需重复添加完整 License：
>
> - C/C++：`// Adapted to tecorigin hardware`
> - Python：`# Adapted to tecorigin hardware`

### 1.3 （可选）优化代码

Python 代码请遵循 [PEP8](https://peps.python.org/pep-0008/) 规范。提交 PR 前，请完整检查代码，确认是否有可以进一步优化的部分（例如删除无关代码等）。可使用 [pylint](https://www.pylint.org/) 等格式化工具统一代码风格。

此外，请严格遵守以下命名与编码规范：

| 规范 | 说明 |
|------|------|
| **文件/文件夹命名** | 使用下划线 `_` 分隔单词，**禁止**使用连字符 `-` |
| **路径拼接** | 必须使用 `os.path.join`，**禁止**使用字符串拼接 |
| **参数管理** | 权重路径、数据集路径、shape 等参数应通过合理传参统一管理 |
| **命名自解释** | 文件和变量名称应能表明其含义 |
| **代码清理** | 删除无关的调试代码和注释 |

---

## 2. 提交 PR

基于您 fork 的 TecovLLM-ModelZoo 仓库，新建 Pull Request 提交内容。关于如何 fork 仓库及提交 PR，请查阅 GitHub 官方文档：[About pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests)。

### 2.1 目标分支

选择目标分支为：`tecorigin/tecovllm-modelzoo:main`。

### 2.2 PR 标题

标题需标注开发者及适配内容，格式如下：

```
【生态活动】元碁智汇·定义未来 - {团队名称} - 模型推理 - {模型名称}
```

> **示例**：`【生态活动】元碁智汇·定义未来 - 某某团队 - 模型推理 - 适配Seed-OSS-36B-Instruct模型`

### 2.3 PR 说明

PR 说明中应包含以下内容：

| 信息项 | 说明 |
|--------|------|
| **软件栈版本** | 在 Docker 环境中执行 `vllm` 命令，以截图方式提供输出的软件版本信息 |
| **源码参考** | 提供源码参考链接及对应的 commit id 或 tag；如无参考源码，请说明原因 |
| **工作目录** | 适配内容的目录结构 |
| **适配内容** | 参考 §1.1 功能实现章节，说明适配了哪些功能 |
| **结果展示** | 包含功能实现中各验证项的测试结果（截图） |
| **README 自测结果** | 确认 README 已通过自测，非开发者也能够按 README 复现此次 PR 的内容 |
