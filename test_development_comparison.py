"""End-to-end declaration and comparison on explicitly labeled synthetic fixtures."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
import torch
from study import parser,run_study,execution_config
from preprocessing import VARIANTS,BASELINE
from provenance import source_identity,environment,identity,atomic_json,read_json
from report_study import render
from compare_development import compare
from artifacts import verify


class DevelopmentComparisonChecks(unittest.TestCase):
    def test_declared_variants_share_pools_and_report_real_input_changes(self):
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC_transform_comparison_') as directory:
            root=Path(directory)
            torch.set_num_threads(2)
            torch.use_deterministic_algorithms(True)
            base=parser().parse_args(['--smoke','--output',str(root/'unused'),'--epochs','1','--seeds','11',
                '--fractions','.005','--permutations','9','--closure-replicates','9','--device','cpu','--telemetry'])
            rule=dict(version=1,dataset_sha256='synthetic_fixture_v1',source=source_identity(),environment=environment('cpu'),
                base_config=execution_config(base),variants=list(VARIANTS),
                locations={name:dict(run=str(root/name),report=str(root/(name+'_report'))) for name in VARIANTS},
                analysis='SYNTHETIC SOFTWARE CHECK of declaration and comparison')
            payload=dict(declaration_id=identity(rule),protocol=rule)
            declaration=root/'declaration.json'
            atomic_json(declaration,payload)
            for variant in VARIANTS:
                args=copy.copy(base)
                args.output=Path(rule['locations'][variant]['run'])
                args.transform=variant
                args.development_declaration=declaration
                with contextlib.redirect_stdout(io.StringIO()):
                    run_study(args)
                render(args.output,Path(rule['locations'][variant]['report']),diagnostic=True)
            result=compare(declaration,root/'comparison')
            self.assertEqual(result['unique_fits'],15)
            self.assertEqual(len(result['records']),15)
            self.assertEqual(result['independent_event_samples'],0)
            self.assertIn('SYNTHETIC SOFTWARE CHECK',(root/'comparison/REPORT.html').read_text())
            verify(root/'comparison',payload['declaration_id'])
            # Valid new preprocessing inputs change both the recorded metrics and
            # generated plots, while every report remains linked to its own run.
            baseline_report=Path(rule['locations'][BASELINE]['report'])
            robust_report=Path(rule['locations']['robust_floor_1e-12']['report'])
            first=read_json(baseline_report/'metrics.json')
            second=read_json(robust_report/'metrics.json')
            self.assertNotEqual([r['threshold'] for r in first],[r['threshold'] for r in second])
            figure='efficiency_f0.005_cut0.05_tuning.png'
            self.assertNotEqual((baseline_report/figure).read_bytes(),(robust_report/figure).read_bytes())
            self.assertNotEqual((baseline_report/'REPORT.html').read_text(),(robust_report/'REPORT.html').read_text())
            # A protocol edit cannot quietly reuse the previously declared fits.
            changed=copy.deepcopy(payload)
            changed['protocol']['base_config']['epochs']=2
            changed['declaration_id']=identity(changed['protocol'])
            atomic_json(root/'changed.json',changed)
            with self.assertRaises(ValueError):
                compare(root/'changed.json',root/'wrong_comparison')


if __name__ == '__main__':
    unittest.main()
