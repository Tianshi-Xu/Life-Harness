# Native benchmark transport

The public task instructions are provided via native_instructions at runtime.
Task environments and evaluation are outside this workspace and immutable.
tools/native.py forwards ordinary named actions/arguments to the native environment.
You may evolve these Python wrappers, add internal tools, middleware, skills, subagents or memory, and revise the system prompt while keeping native instructions.
The RPC bridge and grader are infrastructure and must not be changed.
Do not change LLM configuration/tracing or read hidden grader data.
There are no released Life rules, H2-H5 ownership requirements or single-mechanism limits.
