"""Preserve PTY output when echo and stdout arrive in the same Docker frame.

AgentRL 0.4.0 drops the entire first frame, assuming it contains only input
echo. Disable terminal echo once when opening a shell and retain every output
frame. Commands still run in the original persistent shell with its native
prompt boundary, timeout, container, filesystem and environment budget.
"""
import asyncio


def install_shell_transport(controller):
    from agentrl.worker.environment._const import SHELL_PROMPT_RE
    original_create = controller.create_shell

    async def read_prompt(stream):
        data = bytearray()
        while True:
            message = await stream.read_out()
            if message is None:
                break
            data.extend(message.data)
            if SHELL_PROMPT_RE.search(data):
                break
        return bytes(data)

    async def create_shell(environment_id, shell='/bin/bash --login'):
        await original_create(environment_id, shell)
        stream = controller._shells[environment_id]
        await stream.write_in(b'stty -echo\n')
        await asyncio.wait_for(read_prompt(stream), 5)

    async def execute_shell(environment_id, command, timeout=30):
        if environment_id not in controller._shells:
            await create_shell(environment_id)
        stream = controller._shells[environment_id]
        await stream.write_in(command.encode('utf-8') + b'\n')
        return await asyncio.wait_for(read_prompt(stream), timeout)

    controller.create_shell = create_shell
    controller.execute_shell = execute_shell
