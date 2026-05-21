# 提交PR

在完成模型开发及贡献模型所需文件后，您可以通过PR（Pull Request）将相应内容提交到本仓库。

本文档主要介绍提交PR（Pull Requests）时的规范要求以及方式。请您在提前PR前，按照规范要求对提交内容进行检查，待所有内容符合规范要求后，再提交PR。主要内容如下：

- PR提交规范：介绍规范要求（包括代码、目录等），按照规范要求，检查待提交内容。
- 提交PR：介绍如何提交PR以及填写PR信息。

## 1. PR提交规范

提交PR前的检查工作主要包含以下几个方面：

- 检查路径规范
- 检查文件及目录规范
- 检查功能实现
- 检查注释及License声明
- （可选）优化代码：检查代码格式或内容是否可以进一步优化。

### 1.1 检查模型文件名及路径规范

您提交的模型路径应当为:`TecovLLM/models/{模型名称}`。请参考该格式创建目录，并将模型文件放在该目录下。

- 模型文件名称：模型文件名称必须严格按照模型权重`config.json`配置文件中的 `model_type`参数值命名，命名格式为：`{model_type}.py`

例如，Step-3.5-Flash模型`config.json`文件中`model_type`参数为`step3p5`，则模型路径为`TecovLLM/models/step3p5.py`。

### 1.2 检查功能实现

提交的内容需要实现以下功能：

- 推理服务验证：按适配指南替换模型文件后，TecovLLM服务可正常启动，单样本推理请求返回正常结果。
- 推理精度验证：在太初提供的Docker环境中，使用TecovLLM服务OpenAI接口对Evalscope标准测试数据集完成推理，精度符合要求。

### 1.3 检查注释及License声明

对于代码中重要的部分，需要加入注释介绍功能，帮助使用者快速熟悉代码结构，包括但不仅限于：

- 函数的功能说明，例如：前后处理相关的`resize`或`nms`等。
- `init`、`save`、`load`等io部分代码。
- 模型运行过程中的的关键状态，例如：打印（print）、保存模型等。

为明确代码版权及遵循相关开源协议，您需要在所有完全自主开发的代码文件和头文件内容最上方添加版权声明和开源许可License。

Tecorign ModelZoo提供了C/C++和Python两种版本的License声明，请根据代码语言进行选择：

**说明**：如果原代码文件已有第三方版权声明，为减轻工作量，您可以直接在原文件基础上添加如下声明：``//Adapted to tecorigin hardware``或``# Adapted to tecorigin hardware``。

* **C/C++ License**

  ```
  // BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
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
  // INTERRUPTION)
  // HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
  // STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
  // WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
  // OF SUCH DAMAGE.
  ```
* **Python License**

  ```
  # BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
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
  # INTERRUPTION)
  # HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
  # STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
  # WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
  # OF SUCH DAMAGE.
  ```

### 1.4 （可选）优化代码

Python代码遵循[PEP8](https://peps.python.org/pep-0008/)规范。提交PR之前，请完整检查代码，确认是否有可以进一步优化的代码（例如：删除无关的代码等），从而让代码变得更优雅。可以使用[lint](https://www.pylint.org/)等format工具统一代码格式，使代码更加规整。

此外，相关代码请严格遵守以下命名规范

- 文件和文件夹命名中，使用下划线"_"代表空格，不要使用"-"。
- 类似权重路径、数据集路径、shape等参数，需要通过合理传参统一管理。
- 文件和变量的名称定义过程中需要能够通过名字表明含义。
- 在代码中定义path时，需要使用os.path.join完成，禁止使用string拼接的方式。

## 2. 提交PR

基于您开发环境的TecovLLM-ModelZoo仓库，新建Pull Requests提交内容。关于如何Fork仓库及提交Pull Request，请查阅github官方使用文档：[About pull requests - GitHub Docs](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests?versionId=free-pro-team%40latest&productId=get-started&restPage=using-github)

提交PR时注意以下事项：

- 目标分支选择`tecorigin/tecovllm-modelzoo:main`。
- PR标题：PR标题需要标注开发者及适配的内容，例如：**【生态活动】元碁智汇·定义未来-团队名称-模型推理-在TecovLLM框架上适配Seed-OSS-36B-Instruct模型**。
- PR说明：PR说明应当包含以下内容。

  * 当前适配的软件栈版本：在Docker环境中执行`vllm`命令即可打印当前软件栈版本，以截图的方式提供即可。
  * 源码参考：提供源码参考链接和对应的`commit id`或`tag`，如果无参考源码，请说明。
  * 工作目录：适配内容的目录结构。
  * 适配内容：参考**功能实现**章节，提供适配内容说明。
  * 结果展示：结果展示应包含适配内容中所包含功能的测试结果（截图）。
  * Readme自测结果：确定Readme已经通过自测，非开发者可以通过README运行此次PR内容。
