"""Unit tests for admin HTTP correlation / request IDs."""

from __future__ import annotations

import logging

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from services_python.correlation_id import (
    CORRELATION_ID_HEADER,
    REQUEST_ID_HEADER,
    CorrelationIdAppKey,
    CorrelationIdFilter,
    correlation_id_middleware,
    ensure_correlation_logging,
    get_correlation_id,
    set_correlation_id,
)


@pytest.mark.asyncio
async def test_middleware_mints_and_echoes_request_id():
    async def handler(request: web.Request) -> web.Response:
        assert request[CorrelationIdAppKey]
        return web.json_response({"ok": True, "cid": request[CorrelationIdAppKey]})

    app = web.Application(middlewares=[correlation_id_middleware])
    app.router.add_get("/admin/ping", handler)
    client = TestClient(TestServer(app))
    async with client:
        resp = await client.get("/admin/ping")
        assert resp.status == 200
        data = await resp.json()
        assert data["cid"]
        assert resp.headers[REQUEST_ID_HEADER] == data["cid"]
        assert resp.headers[CORRELATION_ID_HEADER] == data["cid"]


@pytest.mark.asyncio
async def test_middleware_propagates_incoming_request_id():
    async def handler(request: web.Request) -> web.Response:
        return web.json_response({"cid": request[CorrelationIdAppKey]})

    app = web.Application(middlewares=[correlation_id_middleware])
    app.router.add_get("/admin/ping", handler)
    client = TestClient(TestServer(app))
    async with client:
        resp = await client.get(
            "/admin/ping", headers={REQUEST_ID_HEADER: "client-corr-42"}
        )
        assert resp.status == 200
        data = await resp.json()
        assert data["cid"] == "client-corr-42"
        assert resp.headers[REQUEST_ID_HEADER] == "client-corr-42"


@pytest.mark.asyncio
async def test_middleware_accepts_x_correlation_id_alias():
    async def handler(request: web.Request) -> web.Response:
        return web.json_response({"cid": request[CorrelationIdAppKey]})

    app = web.Application(middlewares=[correlation_id_middleware])
    app.router.add_get("/admin/ping", handler)
    client = TestClient(TestServer(app))
    async with client:
        resp = await client.get(
            "/admin/ping", headers={CORRELATION_ID_HEADER: "alias-99"}
        )
        data = await resp.json()
        assert data["cid"] == "alias-99"


def test_correlation_filter_on_log_records():
    ensure_correlation_logging()
    token = set_correlation_id("log-cid-7")
    try:
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello",
            args=(),
            exc_info=None,
        )
        CorrelationIdFilter().filter(record)
        assert record.correlation_id == "log-cid-7"
        assert get_correlation_id() == "log-cid-7"
    finally:
        from services_python import correlation_id as mod

        mod._correlation_id.reset(token)
