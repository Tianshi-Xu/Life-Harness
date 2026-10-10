"""Infrastructure-only chroot for solver code; native grader remains outside."""
import os
import re
import shutil
import subprocess
import sys
import stat
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runtime_rootfs"


def make_source_readable(workspace):
    workspace = Path(workspace)
    for path in [workspace, *workspace.rglob("*")]:
        if not path.is_symlink():
            path.chmod(0o555 if path.is_dir() else 0o444)


def require_regular_workspace(workspace):
    # Never resolve an agent-created absolute symlink on the host while copying
    # out of a chroot: its host target could be a private dataset or credential.
    root = Path(workspace)
    if root.is_symlink() or any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("native transport refuses symlinks in candidate workspace")


def clone_tree(source, target):
    source, target = Path(source), Path(target)
    target.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        dest = target / path.name
        if path.is_symlink():
            if not os.path.lexists(dest):
                dest.symlink_to(os.readlink(path))
            elif not dest.is_symlink() or os.readlink(dest) != os.readlink(path):
                raise RuntimeError(f"unexpected existing isolation symlink: {dest}")
        elif path.is_dir():
            clone_tree(path, dest)
        else:
            copy_link(path, dest)


def copy_link(source, target):
    if Path(target).exists() and os.path.samefile(source, target):
        return str(target)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    return str(target)


def copy_unchanged_or_atomic(source, target):
    """Avoid rewriting shared runtime modules under concurrent episode imports."""
    source, target = Path(source), Path(target)
    if target.is_file() and source.read_bytes() == target.read_bytes():
        return str(target)
    temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(target)
    return str(target)


def prepare_base():
    if (BASE / ".ready").exists():
        return BASE
    BASE.mkdir(exist_ok=True)
    runtime = ROOT / ".venv"
    python = (runtime / "bin/python").resolve()
    python_home = python.parent.parent
    clone_tree(runtime, BASE / "runtime")
    clone_tree(python_home, BASE / str(python_home).lstrip("/"))
    # Copy only dynamic dependencies, not host /etc, datasets or credentials.
    paths = [python]
    paths.extend(runtime.rglob("*.so"))
    paths.extend((python_home / "lib").rglob("*.so"))
    paths.extend(Path(x) for x in ("/bin/bash", "/usr/bin/git", "/usr/bin/env"))
    done = set()
    for binary in paths:
        output = subprocess.run(["ldd", str(binary)], capture_output=True, text=True).stdout
        for name in re.findall(r"(?:=>\s+)?(/[^\s()]+)", output):
            source = Path(name)
            if source in done or not source.is_file():
                continue
            done.add(source)
            target = BASE / name.lstrip("/")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(source, target, follow_symlinks=True)
    for source in ("/bin/bash", "/usr/bin/git", "/usr/bin/env"):
        path = Path(source)
        if path.exists():
            target = BASE / source.lstrip("/")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    (BASE / "etc").mkdir(exist_ok=True)
    for source in ("/etc/resolv.conf", "/etc/hosts", "/etc/nsswitch.conf"):
        if Path(source).is_file():
            shutil.copy2(source, BASE / "etc" / Path(source).name)
    certs = Path("/etc/ssl/certs")
    if certs.exists():
        shutil.copytree(certs, BASE / "etc/ssl/certs", symlinks=False, copy_function=copy_link, dirs_exist_ok=True)
    (BASE / "tmp").mkdir(mode=0o777, exist_ok=True)
    (BASE / "tmp").chmod(0o777)
    (BASE / "dev").mkdir(exist_ok=True)
    # Python uses getrandom(); /dev/null is sufficient for ordinary libraries.
    if not (BASE / "dev/null").exists():
        os.mknod(BASE / "dev/null", 0o20666, os.makedev(1, 3))
    entry = BASE / "runtime/bin/python"
    if not entry.is_symlink():
        raise RuntimeError("unexpected isolated Python entry point")
    entry.unlink()
    entry.symlink_to(str(python))
    (BASE / ".ready").write_text("isolated Python runtime, no benchmark data\n")
    return BASE


def case_root(workspace, trace_dir):
    require_regular_workspace(workspace)
    if not (BASE / ".ready").exists():
        raise RuntimeError("run isolation.prepare_base with the dedicated Python before live evaluation")
    root = BASE
    ensure_shell_utilities()
    original_venv = root / str(ROOT / ".venv").lstrip("/")
    original_venv.parent.mkdir(parents=True, exist_ok=True)
    if not os.path.lexists(original_venv):
        original_venv.symlink_to("/runtime")
    elif original_venv.is_dir() and not original_venv.is_symlink():
        # ldd may already have populated absolute numpy dependency paths here.
        # Keep those libraries, and make installed console-script shebangs valid.
        for name in ("bin", "pyvenv.cfg"):
            alias = original_venv / name
            if not os.path.lexists(alias):
                alias.symlink_to("/runtime/" + name)
        for source in (root / "runtime/lib").glob("python*/site-packages"):
            destination = original_venv / source.relative_to(root / "runtime")
            destination.mkdir(parents=True, exist_ok=True)
            for package in source.iterdir():
                alias = destination / package.name
                if not os.path.lexists(alias):
                    alias.symlink_to("/runtime/" + str(package.relative_to(root / "runtime")))
    adapter = root / "adapter/ahe_native"
    shutil.copytree(ROOT / "ahe_native", adapter, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__"),
                    copy_function=copy_unchanged_or_atomic)
    upstream = ROOT.parent / "agentic-harness-engineering"
    (root / "upstream").mkdir(exist_ok=True)
    copy_unchanged_or_atomic(upstream / "trace_converter.py", root / "upstream/trace_converter.py")
    # The editable adb install points outside the chroot. Supply only its public
    # upstream package under the already-approved adapter import path.
    debugger = upstream / "agents/evolve_agent/skills/agent-debugger-cli/_source/agent_debugger_core"
    if debugger.is_dir():
        shutil.copytree(debugger, root / "adapter/agent_debugger_core", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__"),
                        copy_function=copy_unchanged_or_atomic)
    reservations = root / "uid_reservations"
    reservations.mkdir(mode=0o700, exist_ok=True)
    # Reserve earlier episodes too; exclusive mkdir prevents cross-process UID
    # collisions as formal benchmark runs grow beyond a small smoke.
    if not (reservations / ".seeded").exists():
        for earlier in (root / "episodes").iterdir():
            if re.fullmatch(r"[0-9a-f]{32}", earlier.name):
                try:
                    (reservations / str(100000 + int(earlier.name[:7], 16))).mkdir()
                except FileExistsError:
                    pass
        (reservations / ".seeded").touch()
    while True:
        token = uuid.uuid4().hex
        uid = 100000 + int(token[:7], 16)
        try:
            (reservations / str(uid)).mkdir()
            break
        except FileExistsError:
            continue
    case = root / "episodes" / token
    case.mkdir(parents=True, mode=0o700)
    os.chown(case, uid, uid)
    shutil.copytree(workspace, case / "workspace",
                    ignore=shutil.ignore_patterns(".git", "__pycache__"))
    # Evolve legitimately creates files under umask 077. Its UID differs from
    # the next solver UID: copied private modes must not make generated modules
    # unreadable. The private episode root remains 0700 for the solver alone;
    # candidate source is root-owned/read-only inside that boundary.
    make_source_readable(case / "workspace")
    # The solver cannot write its candidate source or dependency files. Only
    # explicitly session-scoped scratch is mutable, not shared between episodes.
    for directory in (case / "output", case / "workspace/.home", case / "workspace/.scratch"):
        directory.mkdir()
        os.chown(directory, uid, uid)
    short_memory = case / "workspace/ShortTermMEMORY.md"
    if short_memory.exists():
        os.chown(short_memory, uid, uid)
        short_memory.chmod(0o600)
    for name, number in (("urandom", 9), ("random", 8)):
        if not (root / "dev" / name).exists():
            os.mknod(root / "dev" / name, 0o20666, os.makedev(1, number))
    logical = "/episodes/" + token
    return root, case, logical, uid


def ensure_shell_utilities():
    """Minimal original-agent shell infrastructure, never host data mounts."""
    for name in ("ls", "find", "grep", "head", "tail", "sed", "cat", "mkdir",
                 "cp", "mv", "rm", "timeout", "basename", "dirname", "tee",
                 "sort", "uniq", "cut", "wc", "chmod", "touch", "xargs", "tr"):
        source = Path("/usr/bin") / name
        target = BASE / "usr/bin" / name
        if target.exists() or not source.exists():
            continue
        shutil.copy2(source, target)
        output = subprocess.run(["ldd", str(source)], capture_output=True, text=True).stdout
        for filename in re.findall(r"(?:=>\s+)?(/[^\s()]+)", output):
            dependency = Path(filename)
            dest = BASE / filename.lstrip("/")
            if dependency.is_file() and not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dependency, dest)
    shell = BASE / "bin/sh"
    if not os.path.lexists(shell):
        shell.symlink_to("/bin/bash")
    for name, number in (("null", 3), ("urandom", 9), ("random", 8)):
        device = BASE / "dev" / name
        if not device.exists():
            os.mknod(device, 0o20666, os.makedev(1, number))
        info = device.stat()
        if not stat.S_ISCHR(info.st_mode) or info.st_rdev != os.makedev(1, number):
            raise RuntimeError("unexpected isolated device")
        device.chmod(0o666)  # mknod creation permissions were filtered by umask
    results = BASE / "tmp/nexau_bash_tool_results"
    results.mkdir(exist_ok=True)
    os.chown(results, 0, 0)
    results.chmod(0o1777)
    # Original SDK uses this fixed absolute directory. Retain its implementation
    # but protect each old/new output directory from other episode UIDs.
    for child in results.iterdir():
        if child.is_dir():
            child.chmod(0o700)
    tool_outputs = BASE / "tmp/nexau_tool_outputs"
    tool_outputs.mkdir(exist_ok=True)
    os.chown(tool_outputs, 0, 0)
    tool_outputs.chmod(0o1777)
    for child in tool_outputs.iterdir():
        if child.is_file():
            child.chmod(0o600)


def enter(root, working_directory, uid):
    os.umask(0o077)
    os.chroot(root)
    os.chdir(working_directory)
    os.setgroups([])
    os.setgid(uid)
    os.setuid(uid)


if __name__ == "__main__":
    print(prepare_base())
