import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import meta_harness as meta

class MetaTests(unittest.TestCase):
    def test_saved_manifest_reused_without_reproposal(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'prompts').mkdir()
            items=[{'name':'saved','file':'candidates/saved.py'}]
            (root/'prompts/iter_01_pending_eval.json').write_text(json.dumps({'candidates':items}))
            with patch.object(meta.proposer,'run',side_effect=AssertionError('reproposal')):
                self.assertEqual(meta.propose(1,root,argparse.Namespace(candidates_per_iter=1)),items)

    def test_smoke_invokes_register(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'plugin.py';p.write_text('def register():\n    raise RuntimeError("broken register")\n')
            def run(cmd,**kwargs):
                import subprocess
                return subprocess.run([sys.executable,'-c',cmd[-1]],capture_output=True,text=True)
            # Keep the real subprocess.run for the nested interpreter invocation.
            import subprocess
            real=subprocess.run
            def invoke(cmd,**kwargs):return real([sys.executable,'-c',cmd[-1]],capture_output=True,text=True)
            with patch.object(meta.subprocess,'run',side_effect=invoke):
                ok,error=meta.smoke_check(p)
            self.assertFalse(ok)
            self.assertIn('broken register',error)
