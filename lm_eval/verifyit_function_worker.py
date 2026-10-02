"""Data-only function RPC worker; the candidate container never receives tests."""

import json
import math
import re
import sys
from collections import Counter


MAX_BYTES = 1048576


def encode(value, depth=0):
    if depth > 50:
        raise ValueError("value nesting exceeds limit")
    if type(value) is int and value.bit_length() > 4096:
        return {"type": "integer_hex", "value": format(value, "x")}
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is bytes:
        return {"type": "bytes_hex", "value": value.hex()}
    if type(value) is float:
        if math.isfinite(value):
            return value
        return {"type": "float_value", "value": repr(value)}
    if type(value) in (tuple, list, set, frozenset):
        return {
            "type": type(value).__name__,
            "value": [encode(x, depth + 1) for x in value],
        }
    if type(value) is complex:
        return {
            "type": "complex",
            "value": [encode(value.real, depth + 1), encode(value.imag, depth + 1)],
        }
    if type(value) in (dict, Counter):
        return {
            "type": type(value).__name__,
            "value": [
                [encode(k, depth + 1), encode(v, depth + 1)] for k, v in value.items()
            ],
        }
    raise ValueError("unsupported function transport value")


def decode(value, depth=0):
    if depth > 50:
        raise ValueError("value nesting exceeds limit")
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) is not dict or set(value) != {"type", "value"}:
        raise ValueError("invalid typed function value")
    if value["type"] == "bytes_hex":
        text = value["value"]
        if type(text) is not str or re.fullmatch(r"(?:[0-9a-f]{2})*", text) is None:
            raise ValueError("invalid hexadecimal bytes value")
        return bytes.fromhex(text)
    if value["type"] == "integer_hex":
        text = value["value"]
        if type(text) is not str or re.fullmatch(r"-?[0-9a-f]+", text) is None:
            raise ValueError("invalid hexadecimal integer value")
        return int(text, 16)
    if value["type"] == "float_value":
        text = value["value"]
        if type(text) is not str or text not in ("inf", "-inf", "nan"):
            raise ValueError("invalid floating point value")
        return float(text)
    if type(value["value"]) is not list:
        raise ValueError("invalid typed function sequence")
    values = value["value"]
    if value["type"] == "tuple":
        return tuple(decode(x, depth + 1) for x in values)
    if value["type"] == "list":
        return [decode(x, depth + 1) for x in values]
    if value["type"] in ("set", "frozenset"):
        items = [decode(x, depth + 1) for x in values]
        return set(items) if value["type"] == "set" else frozenset(items)
    if value["type"] == "complex":
        components = [decode(x, depth + 1) for x in values]
        if len(components) != 2 or any(type(x) not in (int, float) for x in components):
            raise ValueError("invalid complex transport value")
        return complex(*components)
    if value["type"] in ("dict", "Counter"):
        items = {decode(k, depth + 1): decode(v, depth + 1) for k, v in values}
        return Counter(items) if value["type"] == "Counter" else items
    raise ValueError("unknown function transport type")


def read(stream, max_bytes=MAX_BYTES):
    line = stream.readline(max_bytes + 1)
    if not line or len(line) > max_bytes or not line.endswith(b"\n"):
        raise ValueError("missing or oversized function response")
    return json.loads(line)


def write(stream, value, max_bytes=MAX_BYTES):
    data = json.dumps(value, allow_nan=False).encode() + b"\n"
    if len(data) > max_bytes:
        raise ValueError("oversized function request")
    stream.write(data)
    stream.flush()


def main():
    transport = sys.stdout.buffer
    sys.stdout = sys.stderr
    request = read(sys.stdin.buffer)
    resource_limit = request.get("resource_limit_bytes")
    if resource_limit is not None:
        # Only the Linux execution worker needs this optional Unix module.
        import resource

        if type(resource_limit) is not int or resource_limit <= 0:
            raise ValueError("invalid candidate resource limit")
        for kind in (resource.RLIMIT_AS, resource.RLIMIT_DATA, resource.RLIMIT_STACK):
            resource.setrlimit(kind, (resource_limit, resource_limit))
    scope = {}
    # Candidate execution is confined to its container.
    exec(request["code"], scope)  # noqa: S102
    function = scope[request["entry_point"]]
    max_bytes = request.get("max_bytes", MAX_BYTES)
    if type(max_bytes) is not int or max_bytes not in (MAX_BYTES, 16 * MAX_BYTES):
        raise ValueError("unsupported function frame bound")
    observation = request.get("result_observation", "identity")
    if observation not in ("identity", "bool_or_presence"):
        raise ValueError("unsupported result observation")
    write(transport, {"ready": True})
    while True:
        request = read(sys.stdin.buffer, max_bytes)
        args, kwargs = decode(request)
        result = function(*args, **kwargs)
        if (
            observation == "bool_or_presence"
            and result is not None
            and type(result) is not bool
        ):
            result = "__verifyit_non_none__"
        write(transport, encode(result), max_bytes)


if __name__ == "__main__":
    main()
