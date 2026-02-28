import hashlib
import hmac
import time

from backend.app.config import get_settings


def sign(request_id: str, ttl_seconds: int = 86400) -> str:
    """Return a time-limited HMAC token: '{expiry_hex}.{hmac_hex}'.

    The expiry is baked into the token so validation is stateless.
    Default TTL is 24 hours — callers should pass settings.cache_ttl_seconds
    so the token and the cached result expire together.
    """
    expiry_hex = format(int(time.time()) + ttl_seconds, "x")
    secret = get_settings().result_token_secret.get_secret_value().encode()
    digest = hmac.new(secret, f"{request_id}:{expiry_hex}".encode(), hashlib.sha256).hexdigest()
    return f"{expiry_hex}.{digest}"


def verify(request_id: str, token: str) -> bool:
    """Verify token signature and expiry. Constant-time comparison prevents timing attacks."""
    try:
        expiry_hex, digest = token.split(".", 1)
        if int(expiry_hex, 16) < time.time():
            return False
        secret = get_settings().result_token_secret.get_secret_value().encode()
        expected = hmac.new(secret, f"{request_id}:{expiry_hex}".encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, digest)
    except Exception:
        return False
