"""Public HTTP callbacks, connected to the addresses that were validated."""
from dataclasses import dataclass
import http.client
import ipaddress
import json
import socket
import ssl
import time
from urllib.parse import quote, urlsplit


@dataclass(frozen=True)
class Target:
    scheme: str
    host: str
    port: int
    path: str
    addresses: tuple


def validate(url: str) -> Target:
    if len(url) > 2048 or any(ord(c) <= 32 or ord(c) == 127 for c in url) or "\\" in url:
        raise ValueError("webhook_url must be a valid public http or https URL")
    try:
        parsed = urlsplit(url)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.fragment or "%" in parsed.hostname):
            raise ValueError
        # Require an ASCII hostname (IDNs can use their explicit A-label).
        # IDNA 2003's built-in codec can map a different Unicode name to the
        # certificate-verified hostname, so do not silently perform that map.
        host = parsed.hostname.encode("ascii").decode("ascii")
        port = parsed.port if parsed.port is not None else (443 if parsed.scheme == "https" else 80)
        if not 1 <= port <= 65535:
            raise ValueError
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (ValueError, UnicodeError, OSError) as exc:
        raise ValueError("webhook_url must resolve to a public http or https address") from exc
    if not addresses:
        raise ValueError("webhook_url has no public address")
    for family, _, _, _, address in addresses:
        ip = ipaddress.ip_address(address[0])
        mapped = getattr(ip, "ipv4_mapped", None)
        if (family not in (socket.AF_INET, socket.AF_INET6) or not ip.is_global
                or ip.is_multicast or ip.is_reserved or (mapped and not mapped.is_global)):
            raise ValueError("webhook_url must resolve only to public addresses")
    path = quote(parsed.path or "/", safe="/%:@!$&'()*+,;=-._~")
    if parsed.query:
        path += "?" + quote(parsed.query, safe="/%?:@!$&'()*+,;=-._~")
    return Target(parsed.scheme, host, port, path, tuple(addresses))


def _connect(target: Target, timeout: float):
    # No hostname lookup here: DNS cannot change between validation and connect.
    deadline = time.monotonic() + timeout
    error = None
    for family, kind, proto, _, address in target.addresses:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        sock = socket.socket(family, kind, proto)
        try:
            sock.settimeout(remaining)
            sock.connect(address)
            return sock
        except OSError as exc:
            error = exc
            sock.close()
    raise OSError("could not connect to webhook") from error


class _HTTP(http.client.HTTPConnection):
    def __init__(self, target: Target):
        super().__init__(target.host, target.port, timeout=15)
        self.target = target

    def connect(self):
        self.sock = _connect(self.target, self.timeout)


class _HTTPS(_HTTP):
    def connect(self):
        super().connect()
        # Keep the original host for SNI and certificate verification even
        # though the TCP connection used a pinned numeric address.
        self.sock = ssl.create_default_context().wrap_socket(self.sock, server_hostname=self.host)


def deliver(url: str, payload: dict) -> None:
    # Revalidate at delivery too: a callback may have waited in the queue.
    target = validate(url)
    conn = (_HTTPS if target.scheme == "https" else _HTTP)(target)
    try:
        conn.request("POST", target.path, body=json.dumps(payload).encode("utf-8"), headers={
            "Content-Type": "application/json", "User-Agent": "YTP-Machine-Webhook/1.0"})
        response = conn.getresponse()
        try:
            if not 200 <= response.status < 300:
                # http.client never follows redirects. Do not read an unbounded
                # response body from a caller-controlled callback either.
                raise OSError(f"webhook returned HTTP {response.status}")
        finally:
            response.close()
    finally:
        conn.close()
