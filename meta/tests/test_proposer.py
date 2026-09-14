import pathlib
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import proposer

class ProposerTests(unittest.TestCase):
    def test_tools_are_restricted_not_just_autoapproved(self):
        with patch.object(proposer.shutil, "which", return_value="qoder"), patch.object(proposer.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")) as run:
            proposer.run("write plugin", pathlib.Path("/tmp"))
        cmd = run.call_args.args[0]
        i = cmd.index("--tools") + 1
        self.assertEqual(cmd[i:i+5], list(proposer.DEFAULT_ALLOWED_TOOLS))
        self.assertNotIn("Bash", cmd[i:i+5])
        self.assertNotIn("Agent", cmd[i:i+5])
        self.assertIn("--disallowed-tools", cmd)
