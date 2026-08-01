"""
Unit tests for OAuth 2.0 provider
"""

import time


def test_oauth_provider_import():
    from services_python.oauth_provider import OAuthClient, OAuthProvider

    assert OAuthProvider is not None
    assert OAuthClient is not None


def test_oauth_provider_creation():
    from services_python.oauth_provider import OAuthProvider

    provider = OAuthProvider()
    assert provider is not None
    assert len(provider.clients) > 0


def test_pkce_verifier_generation():
    from services_python.oauth_provider import PKCEParameters

    verifier = PKCEParameters.generate_verifier()
    assert len(verifier) > 0
    challenge = PKCEParameters.create_challenge(verifier)
    assert len(challenge) == 64


def test_client_registration():
    from services_python.oauth_provider import OAuthProvider

    provider = OAuthProvider()
    client = provider.register_client(
        name="Test Client",
        description="Test OAuth client",
        redirect_uris=["http://localhost:8080/callback"],
        allowed_scopes=["read", "write"],
        is_confidential=True,
    )
    assert client.name == "Test Client"
    assert client.client_id.startswith("distribai-")
    assert len(client.client_secret) > 0


def test_authorization_request():
    from services_python.oauth_provider import OAuthProvider, PKCEParameters

    provider = OAuthProvider()
    verifier = PKCEParameters.generate_verifier()
    challenge = PKCEParameters.create_challenge(verifier)
    request = provider.create_authorization_request(
        client_id="distribai-cli",
        redirect_uri="http://localhost:8765/callback",
        scope="jobs:submit nodes:read",
        code_challenge=challenge,
    )
    assert "client_id" in request
    assert request["client_id"] == "distribai-cli"


def test_authorization_code_generation():
    from services_python.oauth_provider import OAuthProvider, PKCEParameters

    provider = OAuthProvider()
    verifier = PKCEParameters.generate_verifier()
    challenge = PKCEParameters.create_challenge(verifier)
    code = provider.authorize(
        client_id="distribai-cli",
        user_id="user123",
        redirect_uri="http://localhost:8765/callback",
        scope="jobs:submit",
        code_challenge=challenge,
    )
    assert len(code) > 0
    assert code in provider.authorization_codes


def test_token_exchange():
    from services_python.oauth_provider import OAuthProvider, PKCEParameters

    provider = OAuthProvider()
    verifier = PKCEParameters.generate_verifier()
    challenge = PKCEParameters.create_challenge(verifier)
    code = provider.authorize(
        client_id="distribai-cli",
        user_id="user123",
        redirect_uri="http://localhost:8765/callback",
        scope="jobs:submit",
        code_challenge=challenge,
    )
    tokens = provider.exchange_code(
        code=code,
        client_id="distribai-cli",
        client_secret="",
        code_verifier=verifier,
        redirect_uri="http://localhost:8765/callback",
    )
    assert "access_token" in tokens
    assert "refresh_token" in tokens
    assert "token_type" in tokens


def test_invalid_code_exchange():
    from services_python.oauth_provider import OAuthProvider

    provider = OAuthProvider()
    result = provider.exchange_code(
        code="invalid-code",
        client_id="distribai-cli",
        client_secret="",
        code_verifier="verifier",
        redirect_uri="http://localhost:8765/callback",
    )
    assert "error" in result


def test_token_introspection():
    from services_python.oauth_provider import OAuthProvider, PKCEParameters

    provider = OAuthProvider()
    verifier = PKCEParameters.generate_verifier()
    challenge = PKCEParameters.create_challenge(verifier)
    code = provider.authorize(
        client_id="distribai-cli",
        user_id="user123",
        redirect_uri="http://localhost:8765/callback",
        scope="jobs:submit",
        code_challenge=challenge,
    )
    tokens = provider.exchange_code(
        code=code,
        client_id="distribai-cli",
        client_secret="",
        code_verifier=verifier,
        redirect_uri="http://localhost:8765/callback",
    )
    introspection = provider.introspect_token(tokens["access_token"])
    assert introspection["active"] is True
    assert "scope" in introspection
    assert introspection["client_id"] == "distribai-cli"


def test_expired_token_introspection():
    from services_python.oauth_provider import OAuthProvider

    provider = OAuthProvider()
    introspection = provider.introspect_token("invalid-token")
    assert introspection["active"] is False


def test_token_validation():
    from services_python.oauth_provider import OAuthProvider, PKCEParameters

    provider = OAuthProvider()
    verifier = PKCEParameters.generate_verifier()
    challenge = PKCEParameters.create_challenge(verifier)
    code = provider.authorize(
        client_id="distribai-cli",
        user_id="user123",
        redirect_uri="http://localhost:8765/callback",
        scope="jobs:submit nodes:read",
        code_challenge=challenge,
    )
    tokens = provider.exchange_code(
        code=code,
        client_id="distribai-cli",
        client_secret="",
        code_verifier=verifier,
        redirect_uri="http://localhost:8765/callback",
    )
    result = provider.validate_token(tokens["access_token"], "jobs:submit")
    assert result is not None
    result = provider.validate_token(tokens["access_token"], "admin")
    assert result is None


def test_token_refresh():
    from services_python.oauth_provider import OAuthProvider, PKCEParameters

    provider = OAuthProvider()
    verifier = PKCEParameters.generate_verifier()
    challenge = PKCEParameters.create_challenge(verifier)
    code = provider.authorize(
        client_id="distribai-cli",
        user_id="user123",
        redirect_uri="http://localhost:8765/callback",
        scope="jobs:submit",
        code_challenge=challenge,
    )
    tokens = provider.exchange_code(
        code=code,
        client_id="distribai-cli",
        client_secret="",
        code_verifier=verifier,
        redirect_uri="http://localhost:8765/callback",
    )
    refresh_token = tokens["refresh_token"]
    new_tokens = provider.refresh_access_token(
        refresh_token=refresh_token,
        client_id="distribai-cli",
        client_secret="",
    )
    assert "access_token" in new_tokens
    assert new_tokens["access_token"] != tokens["access_token"]


def test_cleanup_expired():
    from services_python.oauth_provider import OAuthProvider

    provider = OAuthProvider()
    code = provider.authorize(
        client_id="distribai-cli",
        user_id="user123",
        redirect_uri="http://localhost:8765/callback",
        scope="jobs:submit",
        code_challenge="challenge123",
    )
    assert code in provider.authorization_codes
    provider.authorization_codes[code].expires_at = time.time() - 1
    provider.cleanup_expired()
    assert code not in provider.authorization_codes
