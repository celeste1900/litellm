"""v2-native OBO token exchange: swap the caller's token for an upstream-bound one.

The pure core of the ``token_exchange`` mode. Given the caller's inbound token and the server's
``TokenExchangeConfig``, ``OboTokenExchanger.exchange`` POSTs the grant selected by ``config.profile``
to the configured endpoint and returns the upstream-bound ``access_token`` as a typed ``OAuthToken``,
or a typed ``CredError`` - never a raise (the HTTP edge is the injected ``ExchangeHttpPost``, whose
adapter contains the I/O). Two profiles share this one engine: ``rfc8693`` (the RFC 8693 token-exchange
grant) and ``entra_obo`` (Microsoft Entra On-Behalf-Of, which is the RFC 7523 ``jwt-bearer`` grant);
only the request form differs, so the cache, single-flight, and TTL machinery are dialect-agnostic. The
exchanged token is cached and single-flighted per ``(subject_token, tenant, config, server)`` so a
repeated caller token skips the IdP round-trip and concurrent calls collapse to one exchange, reusing
the shared in-process cache + coordinator foundation. A rotated caller token hashes to a new key and
re-exchanges. Pure v2: no imports from v1.

A missing/expired exchange is an error, never a fall-through to a weaker source (§1.5): the caller
presenting no token is the resolver arm's 401, and an IdP that does not return a usable token is an
``upstream_unavailable`` here.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Awaitable, Callable
from typing import Literal, Protocol

from typing_extensions import assert_never

from litellm._logging import verbose_logger
from litellm.proxy._experimental.mcp_server.outbound_credentials.oauth_token_store import (
    InMemoryTokenCacheBackend,
    InProcessRefreshCoordinator,
    OAuthToken,
    RefreshCoordinator,
    TokenCacheBackend,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.result import (
    Error,
    Ok,
    Result,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.types import (
    CredError,
    ServerSpec,
    TokenExchangeConfig,
)

# A token with no declared expiry is cached for this long; one with an expiry is cached until then
# minus the skew buffer, floored at the minimum. Values mirror v1's MCP_OAUTH2_* constants; the
# composition root injects the configured ones.
_DEFAULT_TTL_SECONDS = 3600.0
_MIN_TTL_SECONDS = 10.0
_EXPIRY_BUFFER_SECONDS = 60.0

_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:token-exchange"
# Microsoft Entra On-Behalf-Of speaks the RFC 7523 jwt-bearer grant, not RFC 8693, and gates delegation
# behind ``requested_token_use=on_behalf_of`` (a Microsoft extension present in neither RFC).
_JWT_BEARER_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:jwt-bearer"
_REQUESTED_TOKEN_USE_OBO = "on_behalf_of"

# The IdP returns an opaque JSON object; the post adapter hands it over untyped and the exchanger
# validates each field, so no Any leaks past this seam (None == any transport/HTTP failure).
ExchangeHttpPost = Callable[[str, "dict[str, str]"], Awaitable["dict[str, object] | None"]]


class SubjectTokenRejected(Exception):
    """The IdP refused to exchange the subject token (an RFC 8693 4xx, e.g. ``invalid_grant``).

    Distinct from a transport / IdP-availability failure, which the post adapter maps to ``None`` ->
    ``upstream_unavailable`` -> 503 (retryable). A rejected subject is the caller's problem, not the
    gateway's, so the arm surfaces it as a non-retryable 401 (the OBO challenge) instead.
    """


class TokenExchanger(Protocol):
    """Exchanges a caller token for an upstream-bound one, per the server's token_exchange config."""

    async def exchange(
        self, subject_token: str, server: ServerSpec, config: TokenExchangeConfig, *, tenant_id: str = ""
    ) -> Result[OAuthToken, CredError]: ...

    async def invalidate(
        self, subject_token: str, server: ServerSpec, config: TokenExchangeConfig, *, tenant_id: str = ""
    ) -> None: ...


def _cache_key(subject_token: str, tenant_id: str, config: TokenExchangeConfig) -> str:
    """Bind the cache entry to the caller token, the tenant, AND the exchange config that minted it.

    A rotated caller token, a different tenant, profile, endpoint, audience, scope, client_id, secret,
    or subject_token_type all change the key, so two tenants behind the same opaque token never share an
    entry and a config change (including a profile flip that alters the wire form) forces a fresh
    exchange instead of serving a token minted for the old config until TTL. Everything is hashed, so no
    secret is held in the key.
    """
    secret = config.client_secret.get_secret_value() if config.client_secret else ""
    material = "\x00".join(
        (
            subject_token,
            tenant_id,
            config.profile,
            config.token_exchange_endpoint or "",
            config.audience or "",
            config.subject_token_type,
            config.client_id or "",
            secret,
            " ".join(config.scopes),
        )
    )
    return hashlib.sha256(material.encode()).hexdigest()


def _parse_expires_in(raw: object) -> int | None:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return int(raw)
    if isinstance(raw, str):
        try:
            return int(float(raw))
        except ValueError:
            return None
    return None


def _rfc8693_form(
    *,
    subject_token: str,
    subject_token_type: str,
    client_id: str,
    client_secret: str,
    audience: str | None,
    scopes: tuple[str, ...],
) -> dict[str, str]:
    return {
        "grant_type": _GRANT_TYPE,
        "subject_token": subject_token,
        "subject_token_type": subject_token_type,
        "client_id": client_id,
        "client_secret": client_secret,
        **({"audience": audience} if audience else {}),
        **({"scope": " ".join(scopes)} if scopes else {}),
    }


def _entra_obo_form(
    *,
    subject_token: str,
    client_id: str,
    client_secret: str,
    scopes: tuple[str, ...],
) -> dict[str, str]:
    # Microsoft Entra On-Behalf-Of (RFC 7523 jwt-bearer, not RFC 8693): the caller's inbound access
    # token rides as ``assertion`` (its ``aud`` must be this gateway's ``client_id``); the target
    # resource is carried in ``scope`` (e.g. api://<app-id>/.default), since Entra has no audience
    # parameter and ignores subject_token_type; ``requested_token_use=on_behalf_of`` is the Microsoft
    # extension that turns the jwt-bearer grant into a delegation. ``scope`` is required, and the
    # exchange precondition rejects an empty one, so it is always present here.
    return {
        "grant_type": _JWT_BEARER_GRANT_TYPE,
        "assertion": subject_token,
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": " ".join(scopes),
        "requested_token_use": _REQUESTED_TOKEN_USE_OBO,
    }


def _build_exchange_form(
    *,
    profile: Literal["rfc8693", "entra_obo"],
    subject_token: str,
    subject_token_type: str,
    client_id: str,
    client_secret: str,
    audience: str | None,
    scopes: tuple[str, ...],
) -> dict[str, str]:
    match profile:
        case "rfc8693":
            return _rfc8693_form(
                subject_token=subject_token,
                subject_token_type=subject_token_type,
                client_id=client_id,
                client_secret=client_secret,
                audience=audience,
                scopes=scopes,
            )
        case "entra_obo":
            return _entra_obo_form(
                subject_token=subject_token,
                client_id=client_id,
                client_secret=client_secret,
                scopes=scopes,
            )
    assert_never(profile)


class OboTokenExchanger:
    """``TokenExchanger`` that runs the profile's OBO grant once per caller token, then caches the result.

    The HTTP post is injected (``None`` on any IdP failure, mirroring v1: a failed exchange is a miss,
    not a 500). The cache and single-flight coordinator default to the in-process foundation; a
    deployment with no shared state needs nothing more (v1's exchanged-token cache is per-process too).
    The clock is injected so TTL/expiry is deterministic in tests.
    """

    def __init__(
        self,
        http_post: ExchangeHttpPost,
        *,
        cache: TokenCacheBackend | None = None,
        coordinator: RefreshCoordinator | None = None,
        clock: Callable[[], float] = time.time,
        default_ttl_seconds: float = _DEFAULT_TTL_SECONDS,
        min_ttl_seconds: float = _MIN_TTL_SECONDS,
        expiry_buffer_seconds: float = _EXPIRY_BUFFER_SECONDS,
    ) -> None:
        self._http_post = http_post
        self._cache: TokenCacheBackend = cache or InMemoryTokenCacheBackend(clock=clock)
        self._coordinator: RefreshCoordinator = coordinator or InProcessRefreshCoordinator()
        self._clock = clock
        self._default_ttl_seconds = default_ttl_seconds
        self._min_ttl_seconds = min_ttl_seconds
        self._expiry_buffer_seconds = expiry_buffer_seconds

    async def exchange(
        self, subject_token: str, server: ServerSpec, config: TokenExchangeConfig, *, tenant_id: str = ""
    ) -> Result[OAuthToken, CredError]:
        endpoint = config.token_exchange_endpoint
        client_id = config.client_id
        client_secret = config.client_secret
        if not endpoint:
            # No endpoint configured and none discoverable: fail closed (412) rather than guess an IdP
            # or fall back to a weaker source. The caller's token is never sent anywhere.
            return Error(
                CredError.of_precondition_required("token exchange endpoint is not configured for this server")
            )
        if not client_id or client_secret is None:
            return Error(CredError.of_misconfigured("token_exchange requires client_id and client_secret"))
        if config.profile == "entra_obo" and not config.scopes:
            # Entra carries the target resource in ``scope`` (api://<app-id>/.default); with no scope the
            # IdP cannot resolve an audience, so fail closed as misconfigured rather than POST a form the
            # IdP will reject.
            return Error(
                CredError.of_misconfigured("entra_obo token exchange requires a scope (e.g. api://<app-id>/.default)")
            )

        cache_key = _cache_key(subject_token, tenant_id, config)
        server_id = server.server_id
        cached = await self._cache.get(cache_key, server_id)
        if cached is not None:
            verbose_logger.debug("MCP token exchange cache hit for server %s", server_id)
            return Ok(cached)

        form = _build_exchange_form(
            profile=config.profile,
            subject_token=subject_token,
            subject_token_type=config.subject_token_type,
            client_id=client_id,
            client_secret=client_secret.get_secret_value(),
            audience=config.audience,
            scopes=config.scopes,
        )

        async def run_exchange() -> OAuthToken | None:
            fresh = await self._cache.get(cache_key, server_id)
            if fresh is not None:
                return fresh
            verbose_logger.debug(
                "Exchanging token for MCP server %s at %s (audience=%s)", server_id, endpoint, config.audience
            )
            body = await self._http_post(endpoint, form)
            if body is None:
                return None
            token = self._token_from_body(body)
            if token is None:
                return None
            await self._cache.set(cache_key, server_id, token, self._ttl_seconds(token))
            verbose_logger.info("Token exchange succeeded for MCP server %s", server_id)
            return token

        async def reread() -> OAuthToken | None:
            return await self._cache.get(cache_key, server_id)

        try:
            token = await self._coordinator.run(cache_key, server_id, refresh=run_exchange, reread=reread)
        except SubjectTokenRejected as rejected:
            # The IdP rejected the subject token (4xx). This is non-retryable: the caller must
            # re-authenticate with the IdP, so it surfaces as a 401 (the OBO challenge), not a 503.
            return Error(CredError.of_unauthorized(str(rejected) or "subject token rejected by the IdP"))
        if token is None:
            return Error(CredError.of_upstream_unavailable("token exchange did not return a usable access token"))
        return Ok(token)

    async def invalidate(
        self, subject_token: str, server: ServerSpec, config: TokenExchangeConfig, *, tenant_id: str = ""
    ) -> None:
        """Drop the cached exchanged token so the next call re-exchanges (e.g. after an upstream 401)."""
        await self._cache.delete(_cache_key(subject_token, tenant_id, config), server.server_id)

    def _token_from_body(self, body: dict[str, object]) -> OAuthToken | None:
        access_token = body.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            return None
        expires_in = _parse_expires_in(body.get("expires_in"))
        expires_at = self._clock() + expires_in if expires_in is not None else None
        return OAuthToken(access_token=access_token, expires_at=expires_at)

    def _ttl_seconds(self, token: OAuthToken) -> float:
        if token.expires_at is None:
            return self._default_ttl_seconds
        lifetime = max(0.0, token.expires_at - self._clock())
        # Floor at min_ttl, but never cache past the token's own expiry: a token whose remaining
        # lifetime is below the buffer (or even below min_ttl) must not be served stale upstream.
        return min(max(lifetime - self._expiry_buffer_seconds, self._min_ttl_seconds), lifetime)
