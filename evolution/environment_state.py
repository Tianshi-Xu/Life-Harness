"""Detached business-state snapshots and compact, replayable local history."""
import copy
import json
from math import isfinite

from evolution.agentbench_hooks_v31.runtime import object_digest


def _read_only(*_args, **_kwargs):
    raise TypeError("business-state snapshot is read-only; use native environment tools")


class ReadOnlyDict(dict):
    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _read_only

    def __deepcopy__(self, memo):
        return self


class ReadOnlyList(list):
    __setitem__ = __delitem__ = __iadd__ = __imul__ = append = clear = extend = insert = pop = remove = reverse = sort = _read_only

    def __deepcopy__(self, memo):
        return self


def read_only(value, previous=None):
    """JSON-compatible frozen containers, independent for each artifact binding."""
    if isinstance(value, dict):
        old = previous if isinstance(previous, ReadOnlyDict) else {}
        items = {key: read_only(item, old.get(key)) for key, item in value.items()}
        if isinstance(previous, ReadOnlyDict) and len(items) == len(old) and all(key in old and item is old[key] for key, item in items.items()):
            return previous
        return ReadOnlyDict(items)
    if isinstance(value, list):
        old = previous if isinstance(previous, ReadOnlyList) else []
        items = [read_only(item, old[i] if i < len(old) else None) for i, item in enumerate(value)]
        if isinstance(previous, ReadOnlyList) and len(items) == len(old) and all(a is b for a, b in zip(items, old)):
            return previous
        return ReadOnlyList(items)
    if value is not None and (type(value) not in (str, bool, int, float) or isinstance(value, float) and not isfinite(value)):
        raise ValueError(f"environment state must contain finite JSON data: received {type(value).__name__}; "
                         "only dict/list/string/bool/int/finite float/null are supported")
    return previous if type(value) is type(previous) and value == previous else value


def snapshot(value):
    if value is None:
        return None
    if type(value) not in (dict, ReadOnlyDict) or set(value) != {"db"} or type(value["db"]) not in (dict, ReadOnlyDict):
        raise ValueError(f"environment state must be None or a business db dictionary: expected {{'db': dict}}; "
                         f"received type={type(value).__name__}, "
                         f"keys={list(value) if isinstance(value, dict) else 'n/a'}")
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("environment state must contain JSON data, not live objects") from exc


def changes(before, after, path=()):
    """Describe changed JSON leaves; do not retain a full DB per solver call."""
    if before is after:
        return []
    if type(before) is type(after) and isinstance(after, dict):
        result = [{"path": [*path, key], "delete": True} for key in before if key not in after]
        for key, value in after.items():
            result.extend(changes(before[key], value, (*path, key)) if key in before
                          else [{"path": [*path, key], "value": copy.deepcopy(value)}])
        return result
    if type(before) is type(after) and isinstance(after, list) and len(before) == len(after):
        return [change for i, (a, b) in enumerate(zip(before, after))
                for change in changes(a, b, (*path, i))]
    return [] if type(before) is type(after) and before == after else [
        {"path": list(path), "value": copy.deepcopy(after)}]


class StateLog:
    def __init__(self):
        self.records, self.identities = [], {}
        self.current, self.index = None, None
        self.view = None

    def record(self, value):
        value = snapshot(value)
        sha = object_digest(value)
        if self.index is not None and self.records[self.index]["sha256"] == sha:
            return self.index
        if sha not in self.identities:
            entry = {"sha256": sha, "state": value} if not self.records else {
                "sha256": sha, "base": self.index, "changes": changes(self.current, value)}
            self.identities[sha] = len(self.records)
            self.records.append(entry)
        self.current, self.index = value, self.identities[sha]
        self.view = read_only(value, self.view)
        return self.index


def apply_changes(value, patch):
    """Copy only changed paths in a frozen view, preserving unmodified branches."""
    def replace(current, path, change):
        if not path:
            if "delete" in change:
                raise ValueError("cannot delete an environment-state root")
            return read_only(snapshot(change["value"]))
        if isinstance(current, dict):
            items, wrap = dict(current), ReadOnlyDict
        elif isinstance(current, list):
            items, wrap = list(current), ReadOnlyList
        else:
            raise ValueError(f"invalid environment-state path: remaining path={path!r}, "
                             f"container type={type(current).__name__}; paths must traverse dicts/lists")
        key = path[0]
        if len(path) > 1:
            items[key] = replace(current[key], path[1:], change)
        elif "delete" in change:
            del items[key]
        else:
            items[key] = read_only(change["value"])
        return wrap(items)

    if not isinstance(patch, list):
        raise ValueError(f"invalid environment-state changes: expected a list, received {type(patch).__name__}")
    for change_index, change in enumerate(patch):
        if (not isinstance(change, dict) or set(change) not in ({"path", "value"}, {"path", "delete"})
                or not isinstance(change["path"], list)
                or any(type(key) not in (str, int) or type(key) is int and key < 0 for key in change["path"])
                or "delete" in change and change["delete"] is not True):
            raise ValueError(f"invalid environment-state change at changes[{change_index}]: expected "
                             "{'path': [string/nonnegative integer, ...], 'value': JSON} or {'path': [...], 'delete': true}")
        value = replace(value, change["path"], change)
    if value is not None and (not isinstance(value, ReadOnlyDict) or set(value) != {"db"} or not isinstance(value["db"], ReadOnlyDict)):
        raise ValueError("environment state must contain only business db")
    return value


class StateReader:
    def __init__(self, records):
        self.records, self.cache = records, {}

    def get(self, index):
        if index is None:
            return None  # Old state-free runs remain explicitly unavailable.
        if type(index) is not int or not 0 <= index < len(self.records):
            raise ValueError(f"invalid environment-state reference: index={index!r}; "
                             f"expected integer in [0,{len(self.records)}) or None for unavailable state")
        chain, position = [], index
        while position not in self.cache:
            if type(position) is not int or not 0 <= position < len(self.records):
                raise ValueError(f"invalid environment-state reference: chain index={position!r}; "
                                 f"expected integer in [0,{len(self.records)})")
            entry = self.records[position]
            if not isinstance(entry, dict):
                raise ValueError(f"invalid environment-state record at index={position}: expected JSON object, "
                                 f"received {type(entry).__name__}")
            chain.append(position)
            if set(entry) == {"sha256", "state"}:
                break
            if set(entry) != {"sha256", "base", "changes"} or type(entry["base"]) is not int or not 0 <= entry["base"] < position:
                raise ValueError(f"environment-state deltas must refer to an earlier record: index={position}, "
                                 f"base={entry.get('base')!r}; expected keys sha256/base/changes and integer 0<=base<index")
            position = entry["base"]
        for position in reversed(chain):
            entry = self.records[position]
            value = (read_only(snapshot(entry["state"])) if "state" in entry else
                     apply_changes(self.cache[entry["base"]], entry["changes"]))
            if object_digest(value) != entry["sha256"]:
                raise ValueError(f"environment-state hash mismatch at index={position}: "
                                 f"expected={entry['sha256']}, actual={object_digest(value)}; "
                                 "recorded evidence is inconsistent; do not invent replacement state")
            self.cache[position] = value
        return self.cache[index]


def resolve(records, index):
    """Return a detached plain-JSON copy for an explicit evidence read."""
    return snapshot(StateReader(records).get(index))
