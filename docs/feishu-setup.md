# 飞书连接指南

LarkSnap 通过飞书自建企业应用实现告警推送和远程指令控制。

| 功能 | 说明 |
|------|------|
| **告警推送** | 检测到目标后，自动将截图和文本消息发送到飞书聊天 |
| **远程指令控制** | 通过飞书聊天发送指令，远程控制 LarkSnap 的启停和状态查询 |

两种功能均基于同一个飞书自建企业应用，配置 `app_id` 和 `app_secret` 后自动启用。告警推送的目标聊天会在你首次给机器人发指令时自动识别，无需手动配置。

---

## 一、创建自建企业应用

1. 登录 [飞书开放平台](https://open.feishu.cn/app)
2. 点击 **创建企业自建应用**
3. 填写应用名称（如 `LarkSnap Detection`），点击创建

---

## 二、获取 App ID 和 App Secret

在应用详情页 → **凭证与基础信息** 中复制：

- **App ID**：`cli_xxxxxxxxxxxxx`
- **App Secret**：`xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

---

## 三、配置应用权限

进入应用 → **权限管理**，搜索并开通以下权限：

| 权限名称 | 权限标识 | 用途 |
|----------|----------|------|
| 获取与发送单聊、群组消息 | `im:message` | 发送消息到群聊 |
| 上传图片 | `im:resource` | 上传检测截图 |
| 读取用户发给机器人的单聊消息 | `im:message.receive_v1` | 接收指令消息 |

点击 **批量开通**，然后进入 **版本管理与发布** 创建版本并提交审核（企业内部应用通常自动通过）。

---

## 四、配置事件订阅（长连接模式）

用于接收飞书消息事件，实现远程指令控制。

1. 进入应用 → **事件与回调** → **事件配置**
2. 添加事件：搜索 **接收消息** (`im.message.receive_v1`)，点击添加
3. 订阅方式选择 **使用长连接接收回调**
4. 保存配置

**长连接模式优势**：
- 无需公网 IP 或域名
- 无需内网穿透
- SDK 内置鉴权与加密，无需处理解密验签
- 本地开发环境即可使用

---

## 五、将应用添加为群机器人（群聊场景）

如果需要将告警推送到群聊，需要将应用添加为群机器人：

1. 在飞书客户端打开目标群聊（或新建一个群聊）
2. 进入群聊 → **设置** → **群机器人** → **添加机器人**
3. 在列表中选择你创建的自建应用
4. 确认应用出现在群机器人列表中

> 如果是私聊场景，直接给机器人发消息即可，无需此步骤。

---

## 六、配置 config.yaml

```yaml
notifier:
  type: feishu
  app_id: "cli_xxxxxxxxxxxxx"
  app_secret: "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
  send_image: true
  message_template: "[LarkSnap] 检测到 {label}，置信度: {confidence:.2%}，时间: {timestamp}"
  retry:
    max_retries: 3
    retry_interval: 5
```

> 只需配置 `app_id` 和 `app_secret`，长连接指令模式自动启用。告警推送的目标聊天在你首次给机器人发指令时自动识别，无需手动配置 `chat_id`。

---

## 七、远程指令控制

在飞书中给机器人发送以下指令（以 `/` 开头）：

| 指令 | 说明 |
|------|------|
| `/start` | 启动检测 |
| `/stop` | 停止检测 |
| `/pause` | 暂停检测 |
| `/resume` | 恢复检测 |
| `/status` | 查询当前状态 |
| `/help` | 查看帮助信息 |

**使用方式**：
- 在群聊中 @机器人 后发送指令，如：`@LarkSnap /start`
- 或直接私聊机器人发送指令，如：`/start`

> 首次发送指令后，该聊天会自动成为告警推送的目标。后续检测到目标时，截图和告警消息会发送到这个聊天。

---

## 八、消息模板自定义

`message_template` 支持以下占位符：

| 占位符 | 说明 | 示例值 |
|--------|------|--------|
| `{label}` | 检测目标类别 | `person` |
| `{confidence}` | 置信度 | `0.87` |
| `{timestamp}` | 检测时间 | `2026-06-04 21:30:00` |
| `{snapshot_path}` | 截图本地路径 | `snapshots/snapshot_20260604_213000.jpg` |

---

## 九、未配置时的行为

当飞书未配置或连接失败时，LarkSnap 不会中断运行，而是：

1. 在本地日志输出 `WARNING` 级别提示
2. 跳过飞书通知
3. 在本地日志中记录检测结果（包含时间、类别、置信度、截图路径）

日志示例：

```
2026-06-04 21:30:00 - larksnap.notifier.feishu - WARNING - Feishu not configured (no app_id/app_secret), skipping notification
2026-06-04 21:30:00 - larksnap.gateway.controller - INFO - DETECTED: person (confidence: 0.87, time: 2026-06-04 21:30:00, snapshot: snapshots/snapshot_20260604_213000.jpg)
```

---

## 十、常见问题

### Q: 发送消息返回 `permission denied`

确认：
1. 应用已开通 `im:message` 权限
2. 应用已添加为群机器人（群聊场景）
3. 应用版本已发布

### Q: 图片上传失败

确认：
1. 应用已开通 `im:resource` 权限
2. 图片文件存在且未损坏

### Q: tenant_access_token 获取失败

确认：
1. `app_id` 和 `app_secret` 正确
2. 应用已发布（未发布的应用无法获取 token）
3. 应用未被禁用

### Q: 长连接指令不生效

确认：
1. `app_id` 和 `app_secret` 已配置
2. 应用已添加 `im.message.receive_v1` 事件订阅
3. 事件订阅方式选择了 **使用长连接接收回调**
4. 应用已发布
5. 指令以 `/` 开头（如 `/start`）
6. 查看日志中是否有 `Feishu WS client connecting...` 提示

### Q: 长连接模式需要公网 IP 吗

不需要。长连接模式由 SDK 主动向飞书服务器建立 WebSocket 连接，无需公网 IP、域名或内网穿透。

### Q: 告警消息发到哪里

告警消息会发送到你首次给机器人发指令的那个聊天（群聊或私聊）。每次发指令时，推送目标会自动更新为当前聊天。
