"""Opt-in local MySQL transport; Docker remains the default.

Use only with a dedicated benchmark MySQL instance. Each episode gets the same
fresh database and cleanup as the Docker transport, using MySQLDatabase itself.
"""
import asyncio
import uuid

from agentrl.worker.environment import create_controller as docker_controller


class NativeMySQLController:
    def __init__(self, delegation, host="127.0.0.1", port=13306, **options):
        if options:
            raise ValueError(f"Unknown native MySQL options: {sorted(options)}")
        if delegation.get_subtypes() != ["mysql"]:
            raise ValueError("Native transport currently supports DBBench only")
        if host not in ("127.0.0.1", "localhost"):
            raise ValueError("Native MySQL must use a dedicated loopback instance")
        self.delegation = delegation
        self.host = host
        self.port = int(port)
        self.sessions = set()

    async def start_session(self, subtype):
        if subtype != "mysql":
            raise ValueError(f"Unsupported native subtype: {subtype}")
        session = str(uuid.uuid4())
        self.sessions.add(session)
        return session, {subtype: session}, {subtype: self.host}

    async def end_session(self, session):
        self.sessions.discard(session)

    async def background_task(self):
        await asyncio.Event().wait()


def create_controller(driver, delegation, **options):
    if driver == "native_mysql":
        return NativeMySQLController(delegation, **options)
    return docker_controller(driver, delegation, **options)
