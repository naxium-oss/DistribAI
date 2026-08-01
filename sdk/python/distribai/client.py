"""
Main client for DistribAI API
"""

from __future__ import annotations

import aiohttp


class DistribAIError(Exception):
    """
    Base exception for DistribAI API errors.

    All specific exceptions inherit from this class.

    Example:
        try:
            await client.jobs.submit(...)
        except DistribAIError as e:
            print(f"Grid error: {e}")
    """

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
    """
    Raised when authentication fails.

    Indicates invalid credentials or expired tokens.
    """

    pass


class InsufficientCreditsError(DistribAIError):
    """
    Raised when credit balance is insufficient for an operation.

    Indicates the node does not have enough credits to complete the requested action.
    """

    pass


class JobNotFoundError(DistribAIError):
    """
    Raised when a requested job is not found.

    Indicates the job ID does not exist or is not accessible.
    """

    pass


class ValidationError(DistribAIError):
    """
    Raised when request validation fails.

    Indicates invalid parameters or malformed requests.
    """

    pass


class DistribAIClient:
    """
    Main client for interacting with the DistribAI API.

    Provides access to jobs, credits, nodes, and voting subsystems.

    Attributes:
        base_url: Base URL of the DistribAI orchestrator
        api_key: API key for authentication (optional)
        session: aiohttp ClientSession for HTTP requests
        jobs: JobsAPI instance for job management
        credits: CreditsAPI instance for credit operations
        nodes: NodesAPI instance for node management
        votes: VotesAPI instance for voting operations

    Example:
        client = DistribAIClient(base_url="http://localhost:8766")
        await client.connect()
        jobs = await client.jobs.list()
        await client.close()
    """

    def __init__(self, base_url: str, api_key: str | None = None):
        """
        Initialize the DistribAI client.

        Args:
            base_url: Base URL of the orchestrator API (e.g., "http://localhost:8766")
            api_key: Optional API key for authentication

        Example:
            >>> client = DistribAIClient(
            ...     base_url="http://localhost:8766",
            ...     api_key="<your-api-key>"
            ... )
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session: aiohttp.ClientSession | None = None
        self._connected = False

        from .credits import CreditsAPI
        from .jobs import JobsAPI
        from .nodes import NodesAPI
        from .votes import VotesAPI

        self.jobs = JobsAPI(self)
        self.credits = CreditsAPI(self)
        self.nodes = NodesAPI(self)
        self.votes = VotesAPI(self)

    async def connect(self) -> None:
        """
        Establish connection to the DistribAI orchestrator.

        Creates the HTTP session and verifies connectivity.

        Raises:
            ConnectionError: If unable to connect to the orchestrator

        Example:
            >>> await client.connect()
            >>> print("Connected to DistribAI")
        """
        if self._connected:
            return

        self.session = aiohttp.ClientSession()
        try:
            await self._request("GET", "/admin/health")
            self._connected = True
        except Exception as e:
            await self.close()
            raise ConnectionError(f"Failed to connect to DistribAI at {self.base_url}: {e}") from e

    async def close(self) -> None:
        """
        Close the connection to the DistribAI orchestrator.

        Closes the HTTP session and cleans up resources.

        Example:
            >>> await client.close()
            >>> print("Disconnected")
        """
        if self.session:
            await self.session.close()
            self.session = None
        self._connected = False

    async def _request(
        self,
        method: str,
        path: str,
        params: dict | None = None,
        json: dict | None = None,
    ) -> dict:
        """
        Make an HTTP request to the DistribAI API.

        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            path: API endpoint path
            params: Query parameters
            json: JSON request body

        Returns:
            Parsed JSON response

        Raises:
            DistribAIError: For API errors
            AuthenticationError: For authentication failures
        """
        if not self.session:
            raise RuntimeError("Client not connected. Call connect() first.")

        url = f"{self.base_url}{path}"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with self.session.request(
            method, url, params=params, json=json, headers=headers
        ) as response:
            if response.status == 401:
                raise AuthenticationError("Authentication failed")
            if response.status >= 400:
                error_data = (
                    await response.json() if response.content_type == "application/json" else {}
                )
                raise DistribAIError(error_data.get("error", f"HTTP {response.status}"))

            return await response.json()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


class Client:
    """
    Main client for interacting with DistribAI.
    This is the primary entry point for the SDK. It provides access to
    all API endpoints through convenient sub-resources.
    Args:
        api_key: Your DistribAI API key
        base_url: API base URL (default: https://api.distribai.io)
        timeout: Request timeout in seconds (default: 30)
    Example:
        >>> client = distribai.Client(api_key="<your_api_key_here>")
        >>>
        >>>
        >>> job = client.jobs.submit(
        ...     model_name="distribai-small",
        ...     dataset="s3://datasets/train.jsonl",
        ...     steps=1000
        ... )
        >>>
        >>>
        >>> balance = client.credits.balance()
        >>> print(f"Available: {balance.confirmed}")
    """

    DEFAULT_BASE_URL = "https://api.distribai.io"

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        timeout: float = 30.0,
    ):
        from .credits import CreditsAPI
        from .jobs import JobsAPI
        from .nodes import NodesAPI
        from .votes import VotesAPI

        self.api_key = api_key
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.timeout = timeout
        self._session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": f"distribai-python/{self._get_version()}",
            },
            timeout=aiohttp.ClientTimeout(total=timeout),
        )
        self.jobs = JobsAPI(self)
        self.credits = CreditsAPI(self)
        self.nodes = NodesAPI(self)
        self.votes = VotesAPI(self)

    def _get_version(self) -> str:
        from . import __version__

        return __version__

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        """
        Make an HTTP request to the API.
        Args:
            method: HTTP method
            path: API path (without base URL)
            **kwargs: Additional arguments for aiohttp
        Returns:
            JSON response as dict
        Raises:
            AuthenticationError: If authentication fails
            DistribAIError: For other API errors
        """
        url = f"{self.base_url}/{path.lstrip('/')}"
        async with self._session.request(method, url, **kwargs) as response:
            if response.status == 401:
                raise AuthenticationError()
            if not response.ok:
                error_data = await response.json()
                raise DistribAIError(
                    error_data.get("error", "Unknown error"),
                    status_code=response.status,
                    error_code=error_data.get("code"),
                )
            return await response.json()

    async def ping(self) -> bool:
        """
        Check if the API is reachable.
        Returns:
            True if API is reachable, False otherwise
        """
        try:
            await self._request("GET", "/health")
            return True
        except Exception:
            return False

    async def close(self):
        await self._session.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
