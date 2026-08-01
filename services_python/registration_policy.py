"""Controls whether public node registration needs a PoC challenge."""

from __future__ import annotations

from services_python.env_bool import env_truthy


def registration_requires_poc() -> bool:
    """If true, reject plain /v1/nodes/register and require the PoC challenge path."""
    explicit = env_truthy("REGISTRATION_REQUIRE_POC")
    if explicit is not None:
        return explicit
    return False
