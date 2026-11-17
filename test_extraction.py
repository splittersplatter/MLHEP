"""Raw HDF5 fixtures test content caching and sparse-jet extraction, never physics claims."""
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import h5py
import numpy as np
from phase1 import extract, validate_features
from provenance import read_json


class ExtractionChecks(unittest.TestCase):
    def fixture(self, root):
        path = root/'SYNTHETIC_particles.h5'
        raw = np.zeros((8, 2101))
        # Two 1700-GeV jets with two constituents each. No four-prong veto.
        raw[:, :12] = [850, 0, 0, 850, .01, .01, 850, 0, np.pi, 850, .01, np.pi-.01]
        raw[:, -1] = np.arange(8) % 2
        with h5py.File(path, 'w') as handle:
            handle.create_dataset('df/block0_values', data=raw)
            handle.create_dataset('df/block0_items', data=np.arange(2101))
        args = SimpleNamespace(data=path, events=8, degree=4, chunk_size=4, cache_dir=root/'cache', start_row=0)
        return path, args

    def test_content_cache_sparse_jets_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, args = self.fixture(root)
            first = extract(args)
            self.assertEqual(first['x'].shape, (8,72))
            self.assertTrue(np.all(first['constituents'] == 2))
            self.assertEqual(first['jet_eta'].shape, (8,2))
            receipt = args.cache_receipt.copy()
            second = extract(args)
            self.assertTrue(np.array_equal(first['x'], second['x']))
            cache = Path(receipt['path'])
            cache.write_bytes(cache.read_bytes()+b'corrupt')
            with self.assertRaises(ValueError):
                extract(args)

    def test_dataset_edit_preserving_size_and_mtime_gets_new_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, args = self.fixture(root)
            extract(args)
            first = args.cache_receipt
            before = path.stat()
            with h5py.File(path, 'r+') as handle:
                handle['df/block0_values'][0,-1] = 1
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            self.assertEqual(before.st_size, path.stat().st_size)
            extract(args)
            second = args.cache_receipt
            self.assertNotEqual(first['metadata']['dataset_sha256'], second['metadata']['dataset_sha256'])
            self.assertNotEqual(first['path'], second['path'])

    def test_signed_fastjet_mass_roundoff_is_retained_but_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            _,args=self.fixture(root)
            data=extract(args)
            meta=args.cache_receipt['metadata']
            data['jet_mass'][0,0]=-2.1579186437577746e-5
            validate_features(data,meta)
            self.assertLess(data['jet_mass'][0,0],0)
            data['jet_mass'][0,0]=-.1
            with self.assertRaises(ValueError):
                validate_features(data,meta)

    def test_wrong_shapes_rows_and_truth_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, args = self.fixture(root)
            data = extract(args)
            meta = args.cache_receipt['metadata']
            for key, value in [('truth', np.full(8,2)), ('row', np.zeros(8,dtype=int)), ('x', np.ones((8,2)))]:
                broken = dict(data, **{key:value})
                with self.assertRaises(ValueError):
                    validate_features(broken, meta)


if __name__ == '__main__':
    unittest.main()
