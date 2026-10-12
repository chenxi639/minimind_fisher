import hashlib,json,tempfile,unittest
from pathlib import Path
from portable.check_assets import check
class AssetTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        (self.root/'asset').write_bytes(b'synthetic')
        self.manifest={'schema_version':1,'experiment_id':'test-only','assets':{'a':{'sha256':hashlib.sha256(b'synthetic').hexdigest(),'role':'fixture'}}}
        self.mapping={'a':'asset'};self.write()
    def write(self):
        (self.root/'manifest.json').write_text(json.dumps(self.manifest));(self.root/'mapping.json').write_text(json.dumps(self.mapping))
    def run_check(self):return check(self.root/'manifest.json',self.root/'mapping.json')
    def test_match(self):self.assertEqual(self.run_check()['status'],'passed')
    def test_corruption(self):
        (self.root/'asset').write_bytes(b'changed')
        with self.assertRaises(ValueError):self.run_check()
    def test_missing(self):
        self.mapping['a']='missing';self.write()
        with self.assertRaises(FileNotFoundError):self.run_check()
    def test_unknown_mapping(self):
        self.mapping['extra']='asset';self.write()
        with self.assertRaises(ValueError):self.run_check()
    def test_version(self):
        self.manifest['schema_version']=2;self.write()
        with self.assertRaises(ValueError):self.run_check()
    def test_unknown_field(self):
        self.manifest['override']=True;self.write()
        with self.assertRaises(ValueError):self.run_check()
