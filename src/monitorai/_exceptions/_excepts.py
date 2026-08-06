from typing import Literal

class MonitorAIError(Exception):
    """Base exception for MonitorAI"""
    pass

class APIError(MonitorAIError):
    """Unexpected API response"""
    pass

class APIConnectionError(MonitorAIError):
    pass

class APIStatusError(APIError):
    """Raised when an API response has a status code of 4xx or 5xx."""
    pass

class APITimeoutError(MonitorAIError):
    pass

class AuthenticationError(MonitorAIError):
    """
    Invalid API Key
    """
    status:Literal[401]=401

class OAuthError(AuthenticationError):
    pass

class BadRequestError(APIStatusError):
    status_code: Literal[400] = 400 


class ConflictError(APIStatusError):
    status_code: Literal[409] = 409  # pyright: ignore[reportIncompatibleVariableOverride]

class InvalidWebhookSignatureError(ValueError):
    """Raised when a webhook signature is invalid, meaning the computed signature does not match the expected signature."""

class InternalServerError(APIStatusError):
    pass


class NetworkError(MonitorAIError):
    """Connection Error"""
    pass

class NotFoundError(APIStatusError):
    status_code: Literal[404] = 404  # pyright: ignore[reportIncompatibleVariableOverride]


class PermissionDeniedError(APIStatusError):
    status_code: Literal[403] = 403  # pyright: ignore[reportIncompatibleVariableOverride]

class RateLimitError(APIStatusError):
    status_code: Literal[429] = 429  # pyright: ignore[reportIncompatibleVariableOverride]


class ServerError(MonitorAIError):
    """Internal server error."""
    pass

class TimeoutError(MonitorAIError):
    """Request timed out."""
    pass

class UnprocessableEntityError(APIStatusError):
    status_code: Literal[422] = 422  # pyright: ignore[reportIncompatibleVariableOverride]

class ValidationError(MonitorAIError):
    pass

class WebSocketQueueFullError(MonitorAIError):
    """Raised when the outgoing WebSocket message queue exceeds its byte-size limit."""

    pass

