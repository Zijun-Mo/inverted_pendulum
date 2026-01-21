# Isaac Lab 项目模板

## 概述

该项目/仓库是一个用于构建基于 Isaac Lab 的项目或扩展的模板。
它允许你在独立环境中开发，而无需在 Isaac Lab 核心仓库内工作。

**主要特性：**

- `隔离` 在 Isaac Lab 核心仓库之外工作，确保你的开发工作自包含。
- `灵活` 该模板可将你的代码作为 Omniverse 扩展运行。

**关键词：** extension, template, isaaclab

## 安装

- 按照 [安装指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) 安装 Isaac Lab。
  我们建议使用 conda 或 uv 安装方式，这样更方便从终端调用 Python 脚本。

- 将该项目/仓库克隆或复制到 Isaac Lab 安装目录之外（即不要放在 `IsaacLab` 目录内）：

- 使用已安装 Isaac Lab 的 Python 解释器，以可编辑模式安装库：

    ```bash
    # 如果 Isaac Lab 未安装在 Python venv 或 conda 中，请用 'PATH_TO_isaaclab.sh|bat -p' 替代 'python'
    python -m pip install -e source/inverted_pendulum
    ```

- 验证扩展是否正确安装：

    - 列出可用任务：

        注意：如果任务名发生变化，可能需要更新 `scripts/list_envs.py` 中的搜索模式 `"Template-"`，以便能够列出。

        ```bash
        # 如果 Isaac Lab 未安装在 Python venv 或 conda 中，请用 'FULL_PATH_TO_isaaclab.sh|bat -p' 替代 'python'
        python scripts/list_envs.py
        ```

    - 运行任务：

        ```bash
        # 如果 Isaac Lab 未安装在 Python venv 或 conda 中，请用 'FULL_PATH_TO_isaaclab.sh|bat -p' 替代 'python'
        python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
        python scripts/skrl/train.py --task=Template-Inverted-Pendulum-v0 --checkpoint=logs/skrl/cartpole_direct/2026-01-20_13-55-57_ppo_torch/checkpoints/best_agent.pt
        ```

    - 使用虚拟 agent 运行任务：

        这些包含输出零动作或随机动作的虚拟 agent，可用于确认环境配置是否正确。

        - 零动作 agent

            ```bash
            # 如果 Isaac Lab 未安装在 Python venv 或 conda 中，请用 'FULL_PATH_TO_isaaclab.sh|bat -p' 替代 'python'
            python scripts/zero_agent.py --task=<TASK_NAME>
            ```
        - 随机动作 agent
**关键词：** extension, template, isaaclab

            ```bash
            # 如果 Isaac Lab 未安装在 Python venv 或 conda 中，请用 'FULL_PATH_TO_isaaclab.sh|bat -p' 替代 'python'
            python scripts/random_agent.py --task=<TASK_NAME>
            ```

### 设置 IDE（可选）

如需配置 IDE，请按以下步骤操作：

- 在 VSCode 中运行任务：按 `Ctrl+Shift+P`，选择 `Tasks: Run Task`，然后在下拉菜单中运行 `setup_python_env`。
  运行时会提示你输入 Isaac Sim 安装路径的绝对路径。

如果执行成功，会在 `.vscode` 目录下创建一个 `.python.env` 文件。
该文件包含 Isaac Sim 与 Omniverse 提供的所有扩展的 python 路径。
这有助于在编写代码时为各模块提供智能索引与提示。

### 作为 Omniverse 扩展进行设置（可选）

我们提供了一个示例 UI 扩展：`source/inverted_pendulum/inverted_pendulum/ui_extension_example.py`，启用扩展后会自动加载。

要启用扩展，请按以下步骤操作：

1. **将此项目/仓库的搜索路径添加到扩展管理器**：
    - 通过 `Window` -> `Extensions` 打开扩展管理器。
    - 点击 **汉堡菜单图标**，进入 `Settings`。
    - 在 `Extension Search Paths` 中输入该项目/仓库 `source` 目录的绝对路径。
    - 如果还未添加，在 `Extension Search Paths` 中输入 Isaac Lab 扩展目录的路径（`IsaacLab/source`）。
    - 点击 **汉堡菜单图标**，再点击 `Refresh`。

2. **搜索并启用你的扩展**：
    - 在 `Third Party` 分类下找到你的扩展。
    - 切换开关以启用扩展。

## 代码格式化

我们提供了 pre-commit 模板来自动格式化代码。
要安装 pre-commit：

```bash
pip install pre-commit
```

然后运行：

```bash
pre-commit run --all-files
```

## 故障排查

### Pylance 缺少扩展索引

在某些 VSCode 版本中，部分扩展的索引可能缺失。
此时可在 `.vscode/settings.json` 中的键 `"python.analysis.extraPaths"` 下添加扩展路径。

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/inverted_pendulum"
    ]
}
```

### Pylance 崩溃

如果 `pylance` 崩溃，通常是索引文件过多导致内存不足。
可以在 `.vscode/settings.json` 中的 `"python.analysis.extraPaths"` 下，注释掉一些不需要的 omniverse 包路径。
下面是一些可能可以排除的示例：

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // 动画相关包
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI 工具
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI 工具
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services 工具
...
```
### 日志曲线
```bash
    tensorboard --logdir logs
```
