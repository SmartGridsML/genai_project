from __future__ import annotations

from ipaddress import ip_address, ip_network

from starlette.requests import Request
from slowapi import Limiter

from backend.app.config import get_settings


def _is_trusted_proxy(client_ip: str | None, trusted_proxies: str) -> bool:
    if not client_ip or not trusted_proxies.strip():
        return False

    for raw in trusted_proxies.split(","):
        entry = raw.strip()
        if not entry:
            continue
        if "/" in entry:
            try:
                if ip_address(client_ip) in ip_network(entry, strict=False):
                    return True
            except ValueError:
                continue
        elif entry == client_ip:
            return True
    return False


def _real_ip(request: Request) -> str:
    settings = get_settings()
    client_ip = request.client.host if request.client else None

    if settings.trust_proxy_headers and _is_trusted_proxy(client_ip, settings.trusted_proxy_ips):
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            forwarded_ip = forwarded.split(",")[0].strip()
            if forwarded_ip:
                return forwarded_ip

    return client_ip or "unknown"


limiter = Limiter(
    key_func=_real_ip,
    storage_uri=get_settings().redis_url,
    in_memory_fallback_enabled=True,
)
