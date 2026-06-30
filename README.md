# LarkSnap

跨平台（Linux / Windows 10+）AI 检测与飞书通知系统，**一套代码**支持三种启动方式：

| 模式 | 命令 | 场景 |
| --- | --- | --- |
| Qt 图形界面 | `larksnap qt` | 现场调试、实时预览 |
| 系统托盘 | `larksnap tray` | 桌面后台运行、菜单控制 |
| 系统服务 | `larksnap service` + `larksnap install` | 开机自启、故障自愈、无界面 |

> 详细平台差异（systemd / SCM、CUDA extras 拆分等）见 [docs/platforms.md](docs/platforms.md)。

---

## 1. 安装

```bash
# 安装 uv（一次性）
curl -LsSf https://astral.sh/uv/install.sh | sh        # Linux/macOS
irm https://astral.sh/uv/install.ps1 | iex             # Windows PowerShell
```

依赖按"**核心 7 项 + 5 类 opt-in extras**"组织。`uv sync` 不带任何 extra 时只装核心（`pydantic / pyyaml / httpx / lark-oapi / pyzmq / numpy / opencv-python`）—— **不包含**推理后端、Qt、托盘、平台服务。明确按场景选 extra：

| 场景 | 命令 |
| --- | --- |
| 桌面 GUI（最常见） | `uv sync --extra qt,tray --extra cpu` |
| Linux 桌面 + CUDA | `uv sync --extra qt,tray --extra gpu-cuda-linux` |
| Windows 桌面 + CUDA | `uv sync --extra qt,tray --extra gpu-cuda-windows` |
| **纯后台服务**（无 Qt） | `uv sync --extra service --extra cpu` |
| Linux 服务 + CUDA | `uv sync --extra service --extra gpu-cuda-linux` |
| Windows 服务 + CUDA | `uv sync --extra service --extra gpu-cuda-windows` |
| 一键全装（按当前平台） | `uv sync --extra all` |
| + 测试 / lint | 上面任一行追加 `--extra dev` |

`service` extra 是跨平台 umbrella —— 在 Windows 自动拉 `pywin32`，在 Linux 自动拉 `sdnotify`。需要精确控制时也可直选 `service-windows` / `service-linux`。

> `uv` 会自动管理 venv、锁定依赖（`uv.lock`）并按平台 marker 拉取正确的 wheel。**CPU 与 GPU extra 互斥**（`onnxruntime` vs `onnxruntime-gpu`），切换时不要同时选。

---

## 2. 配置

复制示例配置后修改：

```bash
cp config/config.example.yaml config/config.yaml     # Linux/macOS
copy config\config.example.yaml config\config.yaml   # Windows
```

最小可用配置只需填 **飞书凭据**（`notifier.app_id` / `notifier.app_secret`），其余字段全部带默认。

### 关键字段速览

```yaml
camera:                  # 摄像头
  device_index: 0        # 设备号，0 = 默认摄像头
  width: 1280 / height: 720 / fps: 30
  capture_interval: 1.0  # 抽帧间隔（秒）

detector:                # 检测器（CPU/CUDA 模型分离）
  type: seg              # 当前支持：seg（实例分割）
  confidence_threshold: 0.5
  target_classes: [person, car, dog]   # 多类监控：列出想监控的 COCO 类别
  seg:
    model_path: models/seg-model.onnx
    provider: cpu        # cpu | cuda（需安装对应 extra）

notifier:                # 飞书机器人
  type: feishu
  app_id: ""             # 在 https://open.feishu.cn/app 创建应用
  app_secret: ""         # 首次给机器人发送 /start 自动获取 chat_id

gateway:                 # 快照保存与通知节流
  notification_interval: 30   # 同一类别最短保存/通知间隔（秒），已重命名为快照的 save_interval；保留字段名以便旧配置兼容
  snapshot_dir: snapshots     # 检测快照保存目录（由快照服务管理）

service:                 # 操作系统服务注册信息
  name: LarkSnap
  user: larksnap         # Linux 下运行用户（systemd 单元会用到）
  windows_start_type: auto   # Windows SCM 启动类型：auto | manual | disabled | delayed
                             # 留空时默认 auto（开机自启）

logging:
  level: INFO
  file_path: logs/larksnap.log
```

> 各字段默认值与说明见 [config/config.example.yaml](config/config.example.yaml) 注释。

---

## 3. 运行

```bash
uv run larksnap qt                 # Qt GUI
uv run larksnap tray               # 托盘（无窗口）
uv run larksnap service            # 前台服务进程（Ctrl+C / SIGTERM 退出）
uv run larksnap -c /path/cfg.yaml qt   # 指定配置文件
```

> **Headless 部署提示**：如果只装了 `service` extra（没装 `qt`），直接 `uv run larksnap`（无子命令）会抛 `ModuleNotFoundError: No module named 'PySide6'`，因为默认子命令是 `qt`。**显式带子命令** `larksnap service` 即可走无 Qt 的服务路径。

## 4. 注册为系统服务

```bash
# Linux（systemd，需 root）
sudo ./scripts/install_service.sh

# Windows（SCM，**必须用管理员 PowerShell**）
.\scripts\install_service.bat
```

服务名 `LarkSnap`。Windows 默认开机自启（`start_type=auto`）；Linux 由 systemd 单元中的 `Restart=always` 实现故障自愈。如需关闭自启，在 `config.yaml` 把 `service.windows_start_type` 改成 `manual` 后重装。

卸载：

```bash
sudo ./scripts/uninstall_service.sh         # Linux
.\scripts\uninstall_service.bat             # Windows
```

管理命令：

```bash
# Linux
sudo systemctl status larksnap
sudo journalctl -u larksnap -f       # 实时日志
sudo systemctl restart larksnap

# Windows
sc query LarkSnap                    # 查状态
sc qc LarkSnap                       # 查启动类型（START_TYPE 行）
sc start LarkSnap                    # 启动
sc stop LarkSnap                     # 停止
```

> **Windows 提权提示**：SCM 注册服务要求 `High Mandatory Level`（管理员），普通终端下 `uv run larksnap install` 会静默失败。**Win+X → Windows PowerShell (管理员)** 后再跑 install 脚本，或 `Start-Process powershell -Verb RunAs` 提权。
>
> **Windows 自启验证**：`sc qc LarkSnap` 输出的 `START_TYPE` 应是 `2   AUTO_START`。如果显示 `3   DEMAND_START`（手动），说明 `service.windows_start_type` 改完没有重装 —— 跑一次 `uv run larksnap install` 即可。

---

## 5. 测试

```bash
uv run pytest                   # 全量
uv run pytest -m "not stress"   # 快速回归（跳过压测）
uv run pytest tests/test_service.py tests/test_main_cli.py   # 跨平台服务层
```

---

## 6. GPU / CPU 环境切换

`uv` 通过 extras 隔离不同推理后端，避免互相污染。**CPU 和 GPU extra 互斥**，切换前要 kill 掉占用 onnxruntime DLL 的 Python 进程（见下方小贴士）。

```bash
# CPU 模式（默认）
uv sync --extra cpu

# GPU 模式：先确认 NVIDIA 驱动 + CUDA 11.8+ 已就位
nvidia-smi                                          # 验证驱动

# 切 GPU 包（卸 onnxruntime，装 onnxruntime-gpu + cu11 工具链）
uv sync --extra gpu-cuda-linux      # Linux
uv sync --extra gpu-cuda-windows    # Windows

# 然后在 config.yaml 里打开 GPU
#   detector.seg.provider: cuda
```

切换不会丢失代码，但会重新解析 lockfile 并装/卸对应包。

### 6.1 后台服务 + GPU（headless 部署）

无 Qt、无桌面环境下的最小部署：

```bash
# 装包：service（pywin32 / sdnotify）+ GPU 后端
uv sync --extra service --extra gpu-cuda-windows    # Windows
uv sync --extra service --extra gpu-cuda-linux      # Linux

# 配置：关掉 CPU 兜底之外，还要把 provider 切到 cuda
# config.yaml:
#   detector:
#     seg:
#       provider: cuda

# 注册为系统服务
sudo ./scripts/install_service.sh                   # Linux
.\scripts\install_service.bat                       # Windows（管理员）

# 验证 GPU 真的在用
sc query LarkSnap | findstr STATE                   # Windows：RUNNING
nvidia-smi                                          # 看 python.exe 是否占显存
journalctl -u larksnap | grep "providers"           # Linux：日志里看 CUDAExecutionProvider
```

> **Windows DLL 锁坑**：在两个 onnxruntime 包之间切换时，如果之前有 Python 进程在跑（包括 IDE 的 Python 内核、之前的 `larksnap service` 实例），`uv sync` 会报 `拒绝访问 (os error 5)`。先：
>
> ```powershell
> Get-Process python, pythonw -ErrorAction SilentlyContinue | Stop-Process -Force
> uv sync --extra gpu-cuda-windows
> ```
>
> 一劳永逸：把 `.venv` 加进 Defender 排除列表 —— `Add-MpPreference -ExclusionPath "E:\Projects\AI\LarkSnap\.venv"`。

---

## 7. 项目结构

```
src/larksnap/
  main.py                 # CLI 入口（qt | tray | service | install | uninstall）
  service/                # 跨平台服务层
    platform_utils.py     #   平台探测
    runner.py             #   共享 ServiceRunner
    linux.py / windows.py #   systemd / SCM 适配
    tray.py               #   托盘
config/config.example.yaml
scripts/                  # 安装/卸载脚本 + systemd unit
tests/                    # pytest（跨平台 + 平台 marker）
docs/platforms.md         # 平台与启动方式选择
```

## License

MIT
