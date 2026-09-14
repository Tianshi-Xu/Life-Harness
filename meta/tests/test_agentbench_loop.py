import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import agentbench_loop as loop
import evidence
from test_evidence import row


class LoopTests(unittest.TestCase):
    def setup_run(self, root):
        src = root / 'source'; domain = src / 'dbbench'
        domain.mkdir(parents=True); (src / 'task_snapshot').mkdir()
        (src / 'task_snapshot' / 'frozen.py').write_text('x=1')
        (src / 'legacy_disabled.py').write_text('')
        for name in ['protocol.py', 'DESIGN_GUIDE.md', 'baseline.py']:
            (domain / name).write_text('# initial')
        evidence.save(src / 'manifest.json', {'dbbench': {'indices': list(range(12))}})
        holdout = root / 'heldout.json'; evidence.save(holdout, [100, 101])
        args = loop.parser().parse_args(['--source', str(src), '--run-dir', str(root/'run'),
               '--domain', 'dbbench', '--baseline', str(domain/'baseline.py'), '--rounds', '3',
               '--screen-size', '4', '--heldout-indices', str(holdout)])
        return args

    def test_prepare_resume_and_frozen_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.setup_run(pathlib.Path(tmp))
            before = loop.prepare(args)
            self.assertEqual(before, loop.prepare(args))
            args.screen_size += 1
            with self.assertRaises(ValueError): loop.prepare(args)
            args.screen_size -= 1
            (args.run_dir / 'task_snapshot/frozen.py').write_text('changed')
            with self.assertRaises(ValueError): loop.prepare(args)

    def test_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.setup_run(pathlib.Path(tmp))
            evidence.save(args.heldout_indices, [1])
            with self.assertRaises(ValueError): loop.prepare(args)

    def test_staged_loop_reject_tie_accept_and_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.setup_run(pathlib.Path(tmp)); phases = []
            def fake_phase(root, domain, name, candidate, ids=None, heldout=False):
                phases.append((name, ids is not None, heldout))
                ids = ids if ids is not None else ([100, 101] if heldout else list(range(12)))
                # Round1 regresses on screen; round2 ties full; round3 improves full.
                values = {str(i): int(i % 3 == 0) for i in ids}
                if name == 'iter_01': values = {k: 0 for k in values}
                if name == 'iter_03': values = {k: 1 for k in values}
                summary = {'n': len(ids), 'successes': sum(values.values()), 'infra_errors': 0,
                           'per_task': values, 'tokens': len(ids)*100, 'plugin_hash': evidence.digest(candidate),
                           'runner_hash': 'r', 'manifest_hash': 'm'}
                path = root/domain/'evals'/name
                for i in ids:
                    evidence.save(path/f'episode_{i}.json', row(i, values[str(i)]))
                evidence.save(path/('screen_summary.json' if ids is not None and len(ids)==4 else 'summary.json'), summary)
                return summary
            def fake_invoke(root, workspace, prompt, tag, model):
                if '_read_' in tag:
                    evidence.save(workspace/'retrieval_request.json', [{'group': 'sql_syntax', 'start_turn': 1, 'end_turn': 1}])
                else:
                    (workspace/'candidate.py').write_text('# candidate ' + tag)
                    evidence.save(workspace/'proposal.json', {'name': tag, 'target_layer':'h2', 'changed_symbols':[],
                                  'hypothesis':'test', 'evidence_tasks':[], 'evidence_groups':['sql_syntax'],
                                  'counterexamples':'success', 'risks':'unknown'})
            with patch.object(loop, 'phase', side_effect=fake_phase), patch.object(loop, 'invoke', side_effect=fake_invoke), patch.object(loop, 'preflight', return_value={'pass': True}):
                loop.run(args)
                state = json.loads((args.run_dir/'dbbench/state.json').read_text())
                self.assertEqual([r['accepted'] for r in state['rounds']], [False, False, True])
                self.assertIsNone(state['rounds'][0]['full_successes'])
                self.assertNotIn(('iter_01', False, False), phases)
                self.assertEqual(state['accepted'], 'candidate_03.py')
                count = len(phases); loop.run(args)
                self.assertEqual(count, len(phases))
                args.finalize = True; loop.run(args)
                self.assertEqual(json.loads((args.run_dir/'dbbench/state.json').read_text())['status'], 'tested')
                args.finalize = False
                with self.assertRaises(ValueError): loop.run(args)
                self.assertFalse(any('100' in p.name or '101' in p.name for p in (args.run_dir/'dbbench/proposer_03').rglob('*')))

    def test_evaluator_screen_cache_is_candidate_specific(self):
        import concurrent.futures
        with tempfile.TemporaryDirectory() as tmp:
            args = self.setup_run(pathlib.Path(tmp)); loop.prepare(args)
            manifest_path = args.run_dir/'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            manifest.update(hashes={}, data_hashes={}, concurrency_per_dataset=1)
            evidence.save(manifest_path, manifest)
            ev = loop.load_evaluator(args.run_dir)
            count = []
            class Pool:
                def __init__(self, **kwargs): pass
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def submit(self, fn, index):
                    count.append(index)
                    f = concurrent.futures.Future(); f.set_result(row(index, index % 3 == 0)); return f
            candidate = args.run_dir/'dbbench/baseline.py'
            ids = args.run_dir/'screen.json'; evidence.save(ids, [0, 1, 2, 3])
            with patch.object(ev, 'validate_plugin'), patch.object(ev, 'ProcessPoolExecutor', Pool):
                ev.evaluate('dbbench', 'iter_01', candidate, ids)
                self.assertFalse((args.run_dir/'dbbench/evals/iter_01/summary.json').exists())
                result = ev.evaluate('dbbench', 'iter_01', candidate)
                self.assertEqual(result['n'], 12)
                self.assertEqual(len(count), 12)
                candidate.write_text('# changed candidate')
                with self.assertRaises(AssertionError):
                    ev.evaluate('dbbench', 'iter_01', candidate)
