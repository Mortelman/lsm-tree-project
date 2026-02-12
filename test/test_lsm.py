import os
import sys
import shutil
import unittest
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from internal.domain import BloomFilter, SSTable, MemTable, Level, LSMTree


class TestBloomFilter(unittest.TestCase):
    
    def test_add_and_contains(self):
        bf = BloomFilter(m=1000, k=5)
        
        bf.add("key1")
        bf.add("key2")
        bf.add("key3")
        
        self.assertIn("key1", bf)
        self.assertIn("key2", bf)
        self.assertIn("key3", bf)
    
    def test_false_negatives(self):
        bf = BloomFilter(m=10000, k=7)
        
        keys = [f"key_{i}" for i in range(1000)]
        for key in keys:
            bf.add(key)
        
        for key in keys:
            self.assertIn(key, bf, f"False negative for {key}")
    
    def test_serialization(self):
        bf = BloomFilter(m=1000, k=5)
        bf.add("test_key")
        
        data = bf.serialize()
        bf2 = BloomFilter.deserialize(data)
        
        self.assertIn("test_key", bf2)
        self.assertEqual(bf.m, bf2.m)
        self.assertEqual(bf.k, bf2.k)
        self.assertEqual(bf.bits, bf2.bits)


class TestMemTable(unittest.TestCase):
    
    def test_put_and_get(self):
        mt = MemTable()
        
        mt.put("key1", "value1")
        mt.put("key2", "value2")
        
        found, value = mt.get("key1")
        self.assertTrue(found)
        self.assertEqual(value, "value1")
        
        found, value = mt.get("key2")
        self.assertTrue(found)
        self.assertEqual(value, "value2")
    
    def test_get_nonexistent(self):
        mt = MemTable()
        
        found, value = mt.get("nonexistent")
        self.assertFalse(found)
        self.assertIsNone(value)
    
    def test_delete(self):
        mt = MemTable()
        
        mt.put("key1", "value1")
        mt.delete("key1")
        
        found, value = mt.get("key1")
        self.assertTrue(found)
        self.assertIsNone(value)
    
    def test_update(self):
        mt = MemTable()
        
        mt.put("key1", "value1")
        mt.put("key1", "value2")
        
        found, value = mt.get("key1")
        self.assertTrue(found)
        self.assertEqual(value, "value2")
    
    def test_scan(self):
        mt = MemTable()
        
        mt.put("a", "1")
        mt.put("b", "2")
        mt.put("c", "3")
        mt.put("d", "4")
        mt.put("e", "5")
        
        results = list(mt.scan("b", "d"))
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0], ("b", "2"))
        self.assertEqual(results[1], ("c", "3"))
        self.assertEqual(results[2], ("d", "4"))
    
    def test_items_sorted(self):
        mt = MemTable()
        
        mt.put("c", "3")
        mt.put("a", "1")
        mt.put("b", "2")
        
        items = mt.items()
        self.assertEqual(items, [("a", "1"), ("b", "2"), ("c", "3")])


class TestSSTable(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_create_and_get(self):
        filename = os.path.join(self.test_dir, "test.sst")
        data = [("key1", "value1"), ("key2", "value2"), ("key3", "value3")]
        
        sst = SSTable(filename, data)
        
        self.assertEqual(sst.get("key1"), "value1")
        self.assertEqual(sst.get("key2"), "value2")
        self.assertEqual(sst.get("key3"), "value3")
    
    def test_get_nonexistent(self):
        filename = os.path.join(self.test_dir, "test.sst")
        data = [("key1", "value1")]
        
        sst = SSTable(filename, data)
        
        self.assertIsNone(sst.get("nonexistent"))
    
    def test_tombstone(self):
        filename = os.path.join(self.test_dir, "test.sst")
        data = [("key1", "value1"), ("key2", None), ("key3", "value3")]
        
        sst = SSTable(filename, data)
        
        self.assertEqual(sst.get("key1"), "value1")
        self.assertIsNone(sst.get("key2"))
        self.assertEqual(sst.get("key3"), "value3")
    
    def test_load_from_disk(self):
        filename = os.path.join(self.test_dir, "test.sst")
        data = [("key1", "value1"), ("key2", "value2")]
        
        sst1 = SSTable(filename, data)
        sst2 = SSTable(filename)
        
        self.assertEqual(sst2.get("key1"), "value1")
        self.assertEqual(sst2.get("key2"), "value2")
    
    def test_scan(self):
        filename = os.path.join(self.test_dir, "test.sst")
        data = [("a", "1"), ("b", "2"), ("c", "3"), ("d", "4"), ("e", "5")]
        
        sst = SSTable(filename, data)
        
        results = list(sst.scan("b", "d"))
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0], ("b", "2"))
        self.assertEqual(results[1], ("c", "3"))
        self.assertEqual(results[2], ("d", "4"))


class TestLevel(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_add_table(self):
        level = Level(0, self.test_dir, max_tables=5)
        
        data = [("key1", "value1"), ("key2", "value2")]
        level.add_table(data)
        
        self.assertEqual(len(level), 1)
    
    def test_get(self):
        level = Level(0, self.test_dir, max_tables=4)
        
        data = [("key1", "value1"), ("key2", "value2")]
        level.add_table(data)
        
        found, value = level.get("key1")
        self.assertTrue(found)
        self.assertEqual(value, "value1")
    
    def test_needs_compaction(self):
        level = Level(0, self.test_dir, max_tables=2)
        
        self.assertFalse(level.needs_compaction())
        
        level.add_table([("key1", "value1")])
        self.assertFalse(level.needs_compaction())
        
        level.add_table([("key2", "value2")])
        self.assertTrue(level.needs_compaction())


class TestLSMTree(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_put_and_get(self):
        with LSMTree(data_dir=self.test_dir, memtable_size=1024) as lsm:
            lsm.put("key1", "value1")
            lsm.put("key2", "value2")
            
            self.assertEqual(lsm.get("key1"), "value1")
            self.assertEqual(lsm.get("key2"), "value2")
    
    def test_get_nonexistent(self):
        with LSMTree(data_dir=self.test_dir) as lsm:
            self.assertIsNone(lsm.get("nonexistent"))
    
    def test_delete(self):
        with LSMTree(data_dir=self.test_dir, memtable_size=1024) as lsm:
            lsm.put("key1", "value1")
            self.assertEqual(lsm.get("key1"), "value1")
            
            lsm.delete("key1")
            self.assertIsNone(lsm.get("key1"))
    
    def test_update(self):
        with LSMTree(data_dir=self.test_dir, memtable_size=1024) as lsm:
            lsm.put("key1", "value1")
            lsm.put("key1", "value2")
            
            self.assertEqual(lsm.get("key1"), "value2")
    
    def test_flush(self):
        with LSMTree(data_dir=self.test_dir, memtable_size=1024*1024) as lsm:
            lsm.put("key1", "value1")
            lsm.flush()
            
            self.assertEqual(lsm.get("key1"), "value1")
            
            stats = lsm.stats()
            self.assertGreater(stats['levels'][0]['tables'], 0)
    
    def test_scan(self):
        with LSMTree(data_dir=self.test_dir, memtable_size=1024) as lsm:
            lsm.put("a", "1")
            lsm.put("b", "2")
            lsm.put("c", "3")
            lsm.put("d", "4")
            lsm.put("e", "5")
            
            results = list(lsm.scan("b", "d"))
            self.assertEqual(len(results), 3)
            self.assertEqual(results[0], ("b", "2"))
            self.assertEqual(results[1], ("c", "3"))
            self.assertEqual(results[2], ("d", "4"))
    
    def test_scan_excludes_tombstones(self):
        with LSMTree(data_dir=self.test_dir, memtable_size=1024) as lsm:
            lsm.put("a", "1")
            lsm.put("b", "2")
            lsm.put("c", "3")
            lsm.delete("b")
            
            results = list(lsm.scan("a", "c"))
            keys = [k for k, v in results]
            self.assertNotIn("b", keys)
    
    def test_many_inserts_trigger_compaction(self):
        with LSMTree(data_dir=self.test_dir, memtable_size=512, level_ratio=2) as lsm:
            for i in range(100):
                lsm.put(f"key_{i:05d}", f"value_{i}" * 10)
            
            lsm.flush()
            
            for i in range(100):
                value = lsm.get(f"key_{i:05d}")
                self.assertEqual(value, f"value_{i}" * 10)


if __name__ == "__main__":
    unittest.main()