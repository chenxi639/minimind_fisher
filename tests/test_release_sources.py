import json,tempfile,unittest
from pathlib import Path
from scripts.release_sources import select_sources
class SourceSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        (self.root/'scripts').mkdir();(self.root/'scripts/a.py').write_text('pass');(self.root/'scripts/not_listed.py').write_text('pass')
        self.manifest=self.root/'sources.json'
    def choose(self,names,version=1):
        self.manifest.write_text(json.dumps({'schema_version':version,'research_scripts':names}));return select_sources(self.root,self.manifest)
    def test_explicit_only(self):self.assertEqual(self.choose(['scripts/a.py']),[self.root/'scripts/a.py'])
    def test_index_independence(self):
        before=self.choose(['scripts/a.py']);(self.root/'.git').mkdir()
        (self.root/'.git/index').write_bytes(b'index fixture, not a real Git index')
        self.assertEqual(before,self.choose(['scripts/a.py']))
    def test_missing(self):
        with self.assertRaises(ValueError):self.choose(['scripts/missing.py'])
    def test_duplicate(self):
        with self.assertRaises(ValueError):self.choose(['scripts/a.py','scripts/a.py'])
    def test_traversal(self):
        with self.assertRaises(ValueError):self.choose(['scripts/../outside.py'])
    def test_version(self):
        with self.assertRaises(ValueError):self.choose(['scripts/a.py'],2)
