"""Fail-fast compatibility names for frozen benchmark Task imports; no legacy runtime."""
import sys,types
class DisabledConfig:
 def __init__(self,**kwargs):
  self.enabled=False
  self.h2_enabled=self.h3_enabled=self.h4_enabled=self.h5_enabled=False
class UnavailableRuntime:
 def __init__(self,*args,**kwargs):raise RuntimeError("Legacy runtime must not execute in unified format")
def unavailable(*args,**kwargs):raise RuntimeError("Legacy harness function must not execute")
def install():
 package=types.ModuleType('src.server.harness');package.__path__=[]
 for d,prefix in [('alfworld','ALFWorld'),('dbbench','DBBench')]:
  module=types.ModuleType('src.server.harness.'+d)
  for target in [package,module]:
   target.__dict__[prefix+'HarnessConfig']=DisabledConfig
   target.__dict__[prefix+'HarnessRuntime']=UnavailableRuntime
  module.rescue_tool_call_from_text=unavailable
  sys.modules[module.__name__]=module
 for name in ['patch_take_action_tool_description','patch_dbbench_tool_descriptions','patch_dbbench_system_prompt']:package.__dict__[name]=unavailable
 sys.modules[package.__name__]=package
