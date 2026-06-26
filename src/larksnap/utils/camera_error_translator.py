"""Translate raw OpenCV/MSMF errors to user-friendly Chinese messages.

OpenCV and Windows MSMF produce noisy English/technical errors that are
unhelpful for end users. This module inspects the error text and returns
a clear, actionable message in Chinese.
"""

from __future__ import annotations

import re


# Windows MSMF error codes (HRESULT) → description
MSMF_ERROR_CODES: dict[str, str] = {
    "-1072875772": "摄像头正被其他程序占用",
    "0x80070005": "访问摄像头被拒绝（权限不足）",
    "0x80070490": "设备不存在或索引无效",
    "0xC00D36B4": "MSMF 摄像头初始化失败",
    "0xC00D3704": "MSMF 摄像头读取失败",
}

# OpenCV backend error patterns → description
# Use regex patterns to catch partial matches
OPENCV_ERROR_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"camera index out of range", re.IGNORECASE), "设备索引超出范围"),
    (re.compile(r"obsensor.*index", re.IGNORECASE), "设备索引超出范围"),
    (re.compile(r"failed to open camera", re.IGNORECASE), "打开摄像头失败"),
    (re.compile(r"failed to read frame", re.IGNORECASE), "读取视频帧失败"),
    (re.compile(r"could not find a camera", re.IGNORECASE), "未找到可用的摄像头"),
    (re.compile(r"your device does not have a camera", re.IGNORECASE), "设备没有可用的摄像头"),
    (re.compile(r"permission.*denied", re.IGNORECASE), "摄像头权限被拒绝"),
    (re.compile(r"backend", re.IGNORECASE), "摄像头后端初始化失败"),
    # Memory layout / buffer size issues from internal OpenCV
    (re.compile(r"_step\s*>=\s*minstep", re.IGNORECASE), "视频帧数据不完整或损坏"),
    (re.compile(r"cv::Mat::Mat", re.IGNORECASE), "视频帧数据结构异常"),
    (re.compile(r"assertion failed", re.IGNORECASE), "摄像头数据校验失败"),
    (re.compile(r"unknown exception|server.*exception", re.IGNORECASE), "摄像头驱动异常"),
    (re.compile(r"backend context", re.IGNORECASE), "摄像头后端上下文错误"),
    (re.compile(r"timeout", re.IGNORECASE), "摄像头操作超时"),
]


def translate_camera_error(raw_error: str) -> str:
    """Translate a raw camera error message to a user-friendly Chinese message.

    Args:
        raw_error: Original error string from OpenCV/MSMF.

    Returns:
        A concise, actionable error description in Chinese.
    """
    if not raw_error:
        return "摄像头发生未知错误"

    error_lower = raw_error.lower()
    matched: list[str] = []

    # 1) Check MSMF HRESULT codes
    for code, desc in MSMF_ERROR_CODES.items():
        if code in raw_error or code.lower() in error_lower:
            matched.append(desc)
            break

    # 2) Check OpenCV patterns
    for pattern, desc in OPENCV_ERROR_PATTERNS:
        if pattern.search(error_lower):
            matched.append(desc)
            break

    if matched:
        return " / ".join(matched)

    # 3) Fallback: if it's mostly English with hex/error noise, hide the details
    if any(c in raw_error for c in ["@", "0x", "HRESULT", ".cpp:", "Assertion failed"]):
        return "摄像头硬件或驱动异常"

    # 4) Otherwise return the original text (it might already be informative)
    return raw_error


def is_camera_data_corruption(raw_error: str) -> bool:
    """Check whether the error indicates corrupted/incomplete frame data."""
    error_lower = raw_error.lower()
    return (
        "_step" in error_lower
        and "minstep" in error_lower
    ) or "视频帧数据" in raw_error


def is_camera_in_use_error(raw_error: str) -> bool:
    """Check whether the error indicates camera is occupied by another process."""
    error_lower = raw_error.lower()
    return (
        "-1072875772" in raw_error
        or "0x80070005" in raw_error
        or "占用" in raw_error
        or "in use" in error_lower
        or "access denied" in error_lower
    )


def is_camera_index_error(raw_error: str) -> bool:
    """Check whether the error indicates an invalid camera index."""
    error_lower = raw_error.lower()
    return (
        "index out of range" in error_lower
        or "out of range" in error_lower
        or "索引超出" in raw_error
        or "0x80070490" in raw_error
    )


def get_solution_hint(raw_error: str) -> str:
    """Return a brief solution hint based on the error type."""
    if is_camera_in_use_error(raw_error):
        return (
            "解决方案：\n"
            "1. 关闭其他正在使用摄像头的程序（如视频会议、浏览器等）\n"
            "2. 等待几秒后重试\n"
            "3. 在 Camera 菜单中尝试其他设备索引"
        )
    if is_camera_index_error(raw_error):
        return (
            "解决方案：\n"
            "1. 打开 Camera → Refresh Devices 重新扫描可用摄像头\n"
            "2. 在 Camera → Select Device 中选择另一个设备索引\n"
            "3. 检查设备管理器中摄像头是否正常"
        )
    if is_camera_data_corruption(raw_error):
        return (
            "解决方案：\n"
            "1. 关闭摄像头后重新打开（Camera → Close Camera → Open Camera）\n"
            "2. 检查摄像头分辨率配置是否超出硬件支持范围\n"
            "3. 重启应用程序以重置摄像头驱动"
        )
    return (
        "解决方案：\n"
        "1. 检查摄像头是否正确连接\n"
        "2. 检查驱动程序是否安装\n"
        "3. 尝试重启应用程序"
    )
