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

# 拉取依赖（按需选择 extras）
uv sync                          # 默认：CPU + 托盘（无 Qt/服务）
uv sync --extra qt               # + Qt 图形界面
uv sync --extra gpu-cuda-linux   # Linux + CUDA 12 推理
uv sync --extra gpu-cuda-windows # Windows + CUDA 12 推理
uv sync --extra all              # 全部能力（按当前平台）
uv sync --extra dev              # + 测试/lint
```

`uv` 会自动管理 venv、锁定依赖（`uv.lock`）并按平台 marker 拉取正确的 wheel。

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

## 4. 注册为系统服务

```bash
# Linux（systemd，需 root）
sudo ./scripts/install_service.sh

# Windows（SCM，需管理员 PowerShell）
.\scripts\install_service.bat
```

服务名 `LarkSnap`，开机自启、失败重启。卸载：

```bash
sudo ./scripts/uninstall_service.sh         # Linux
.\scripts\uninstall_service.bat             # Windows
```

管理命令：

```bash
sudo systemctl status larksnap               # Linux
sc query LarkSnap                            # Windows
```

---

## 5. 测试

```bash
uv run pytest                   # 全量
uv run pytest -m "not stress"   # 快速回归（跳过压测）
uv run pytest tests/test_service.py tests/test_main_cli.py   # 跨平台服务层
```

---

## 6. GPU / CPU 环境切换

`uv` 通过 extras 隔离不同推理后端，避免互相污染：

```bash
uv sync --extra gpu-cuda-linux --extra qt      # Linux CUDA 12 + GUI
uv sync --extra gpu-cuda-windows --extra qt    # Windows CUDA 12 + GUI
uv sync --extra cpu --extra tray              # 纯 CPU + 托盘（轻量部署）
```

切换不会丢失代码，但会重新解析 lockfile 并装/卸对应包。

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
