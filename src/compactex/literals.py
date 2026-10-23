"""Deterministic literal pools. Train/dev/sealed pools are disjoint by construction.

Small closed spaces (HTTP status codes, ports) are shared across pools — documented,
accepted; disjointness applies where the value grammar allows it.
"""

import random
from collections.abc import Callable

from compactex.schemas import LiteralKind

POOL_NAMES = ("train", "dev", "sealed")

# Value-grammar pool markers (disjoint by construction)
_NAMED_CODE_FIRST = {"train": "ABCDEFGHI", "dev": "KLMNOPQR", "sealed": "STUVWXYZ"}
_VERSION_MAJOR = {"train": range(1, 12), "dev": range(12, 28), "sealed": range(28, 48)}
_UUID_FIRST = {"train": "012345", "dev": "6789ab", "sealed": "cdef"}
_IP_FIRST_OCTET = {"train": range(10, 90), "dev": range(100, 170), "sealed": range(170, 220)}
_HOST_WORDS = {
    "train": ["billing", "auth", "payment", "gateway", "search", "media", "cache", "notify"],
    "dev": ["delivery", "storage", "crawler", "scheduler", "profiler", "tracker", "queue"],
    "sealed": ["replica", "insight", "ledger", "orbit", "vertex", "prism", "atlas"],
}
_HOST_ZONE = {"train": ["prod", "stage"], "dev": ["dev", "test"], "sealed": ["core", "edge"]}
_PATH_ROOT = {
    "train": ["app", "srv", "opt"],
    "dev": ["mnt", "data", "work"],
    "sealed": ["net", "hub", "sys"],
}
_RES_PREFIX = {
    "train": ["vm", "bucket", "instance"],
    "dev": ["queue", "cluster", "node"],
    "sealed": ["shard", "realm", "plane"],
}
_TICKET_PREFIX = {"train": "INC", "dev": "REQ", "sealed": "OPS"}
_MONTH = {"train": range(1, 5), "dev": range(5, 9), "sealed": range(9, 13)}

# Shared closed spaces (documented non-disjoint)
HTTP_CODES = ["400", "401", "403", "404", "409", "418", "429", "500", "502", "503", "504"]
PORTS = ["22", "80", "443", "8080", "8443", "9090", "5432", "6379", "9200", "27017"]


def _hex(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("0123456789abcdef") for _ in range(n))


def gen_error_code(pool: str, rng: random.Random) -> str:
    if rng.random() < 0.5:
        return rng.choice(HTTP_CODES)
    letters = _NAMED_CODE_FIRST[pool]
    word = "".join(rng.choice(letters) if i == 0 else rng.choice(letters) for i in range(3))
    return f"{word}{rng.randint(10, 99)}-{rng.randint(100, 999)}"


def gen_version(pool: str, rng: random.Random) -> str:
    v = f"{rng.choice(_VERSION_MAJOR[pool])}.{rng.randint(0, 30)}.{rng.randint(0, 40)}"
    if rng.random() < 0.25:
        v += rng.choice(["-rc1", "-rc2", "-beta3", "-alpha1", "-build.7"])
    return "v" + v if rng.random() < 0.4 else v


def gen_uuid(pool: str, rng: random.Random) -> str:
    first = rng.choice(_UUID_FIRST[pool])
    rest = [_hex(rng, 7), _hex(rng, 4), _hex(rng, 4), _hex(rng, 4), _hex(rng, 12)]
    return f"{first}{rest[0]}-{rest[1]}-{rest[2]}-{rest[3]}-{rest[4]}"


def gen_ip(pool: str, rng: random.Random) -> str:
    return ".".join(
        [str(rng.choice(list(_IP_FIRST_OCTET[pool]))), str(rng.randint(0, 255)),
         str(rng.randint(0, 255)), str(rng.randint(1, 254))]
    )


def gen_host(pool: str, rng: random.Random) -> str:
    word = rng.choice(_HOST_WORDS[pool])
    zone = rng.choice(_HOST_ZONE[pool])
    return f"{word}-{zone}-{rng.randint(1, 99):02d}" if rng.random() < 0.7 else f"{word}-{zone}"


def gen_port(pool: str, rng: random.Random) -> str:  # pool unused: shared space
    return rng.choice(PORTS)


def gen_path(pool: str, rng: random.Random) -> str:
    root = rng.choice(_PATH_ROOT[pool])
    depth = rng.randint(1, 3)
    parts = [_hex(rng, rng.randint(3, 6)) if rng.random() < 0.3 else f"seg{rng.randint(1, 99)}"
             for _ in range(depth)]
    return "/" + "/".join([root, *parts])


def gen_resource_id(pool: str, rng: random.Random) -> str:
    return f"{rng.choice(_RES_PREFIX[pool])}-{_hex(rng, rng.randint(4, 8))}"


def gen_ticket_ref(pool: str, rng: random.Random) -> str:
    return f"{_TICKET_PREFIX[pool]}-{rng.randint(10000, 999999)}"


def gen_datetime(pool: str, rng: random.Random) -> str:
    return f"2026-{rng.choice(list(_MONTH[pool])):02d}-{rng.randint(1, 28):02d}"


def gen_url(pool: str, rng: random.Random) -> str:
    return f"https://{gen_host(pool, rng)}{'/' + _hex(rng, 5) if rng.random() < 0.5 else ''}"


GENERATORS: dict[LiteralKind, Callable[[str, random.Random], str]] = {
    "error_code": gen_error_code,
    "version": gen_version,
    "uuid": gen_uuid,
    "ip": gen_ip,
    "url": gen_url,
    "host": gen_host,
    "port": gen_port,
    "path": gen_path,
    "resource_id": gen_resource_id,
    "ticket_ref": gen_ticket_ref,
    "datetime": gen_datetime,
}

PLACEHOLDER_BY_KIND: dict[LiteralKind, str] = {
    "error_code": "ERR",
    "version": "VER",
    "uuid": "UID",
    "ip": "IP",
    "url": "URL",
    "host": "HST",
    "port": "PRT",
    "path": "PTH",
    "resource_id": "RES",
    "ticket_ref": "TKT",
    "datetime": "DTE",
}


def pool_of_value(value: str) -> str | None:
    """Reverse-map a literal to its pool marker (None for shared spaces)."""
    import re

    if re.fullmatch(r"[A-Z]{3}\d{2}-\d{3}", value):
        return next(p for p, letters in _NAMED_CODE_FIRST.items() if value[0] in letters)
    if re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", value):
        return next(p for p, chars in _UUID_FIRST.items() if value[0] in chars)
    # ip BEFORE version: 10.32.4.117 must not be classified as a version
    if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", value):
        return next(p for p, r in _IP_FIRST_OCTET.items() if int(value.split(".")[0]) in r)
    if re.fullmatch(r"v?\d+\.\d+\.\d+.*", value):
        major = int(re.match(r"v?(\d+)", value).group(1))  # type: ignore[union-attr]
        return next(p for p, r in _VERSION_MAJOR.items() if major in r)
    if re.fullmatch(r"(INC|REQ|OPS)-\d+", value):
        return {"INC": "train", "REQ": "dev", "OPS": "sealed"}[value[:3]]
    m = re.fullmatch(r"(\w+)-(\w+)(?:-\d{2})?", value)
    if m and m.group(1) in {w for ws in _HOST_WORDS.values() for w in ws}:
        return next(p for p, ws in _HOST_WORDS.items() if m.group(1) in ws)  # type: ignore[return-value]
    if value.startswith("2026-"):
        return next(p for p, r in _MONTH.items() if int(value[5:7]) in r)
    return None
