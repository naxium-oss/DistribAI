"""
Custom exceptions for DistribAI SDK
"""


class DistribAIError(Exception):
    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        error_code: str | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code


class AuthenticationError(DistribAIError):
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, status_code=401, error_code="AUTH_FAILED")


class RateLimitError(DistribAIError):
    def __init__(self, message: str = "Rate limit exceeded", retry_after: int | None = None):
        super().__init__(message, status_code=429, error_code="RATE_LIMITED")
        self.retry_after = retry_after


class JobNotFoundError(DistribAIError):
    def __init__(self, job_id: str):
        super().__init__(f"Job {job_id} not found", status_code=404, error_code="JOB_NOT_FOUND")
        self.job_id = job_id


class InsufficientCreditsError(DistribAIError):
    def __init__(self, required: float, available: float):
        super().__init__(
            f"Insufficient credits: required {required}, available {available}",
            status_code=402,
            error_code="INSUFFICIENT_CREDITS",
        )
        self.required = required
        self.available = available


class ValidationError(DistribAIError):
    def __init__(self, message: str):
        super().__init__(message, status_code=422, error_code="VALIDATION_ERROR")


class ServerError(DistribAIError):
    def __init__(self, message: str = "Internal server error"):
        super().__init__(message, status_code=500, error_code="SERVER_ERROR")
