class LarkSnapError(Exception):
    """Base exception for LarkSnap application."""

    pass


class CameraError(LarkSnapError):
    """Exception raised for camera-related errors."""

    pass


class DetectorError(LarkSnapError):
    """Exception raised for detector-related errors."""

    pass


class NotifierError(LarkSnapError):
    """Exception raised for notifier-related errors."""

    pass


class ConfigError(LarkSnapError):
    """Exception raised for configuration-related errors."""

    pass


class GatewayError(LarkSnapError):
    """Exception raised for gateway-related errors."""

    pass
