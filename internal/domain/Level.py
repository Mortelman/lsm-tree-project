import os
import heapq
from typing import Optional, List, Tuple, Iterator, Union
from .SSTable import SSTable
from .BloomFilter import BloomFilter


class Level:

    def __init__(self, level_num: int, base_dir: str, max_tables: int = 10):
        self.level_num = level_num
        self.base_dir = base_dir
        self.max_tables = max_tables
        self.tables: List[SSTable] = []
        self._table_counter = 0
        
        self.level_dir = os.path.join(base_dir, f"level_{level_num}")
        os.makedirs(self.level_dir, exist_ok=True)
        
        self._load_existing_tables()

    def _load_existing_tables(self) -> None:
        if not os.path.exists(self.level_dir):
            return
            
        for filename in sorted(os.listdir(self.level_dir)):
            if filename.endswith('.sst'):
                filepath = os.path.join(self.level_dir, filename)
                try:
                    table = SSTable(filepath)
                    self.tables.append(table)
                    try:
                        num = int(filename.replace('.sst', '').split('_')[-1])
                        self._table_counter = max(self._table_counter, num + 1)
                    except ValueError:
                        pass
                except Exception as e:
                    print(f"Warning: Failed to load SSTable {filepath}: {e}")

    def add_table(self, data: Union[List[Tuple[str, Optional[str]]], Iterator[Tuple[str, Optional[str]]]]) -> SSTable:
        filename = os.path.join(self.level_dir, f"table_{self._table_counter}.sst")
        self._table_counter += 1
        
        if not isinstance(data, list):
            data = list(data)
        
        table = SSTable(filename, data)
        self.tables.append(table)
        return table

    def needs_compaction(self) -> bool:
        return len(self.tables) >= self.max_tables

    def get(self, key: str) -> Tuple[bool, Optional[str]]:
        for table in reversed(self.tables):
            value = table.get(key)
            if value is not None:
                return (True, value)
            
            if key in table.bloom:
                # 1. tombstone
                # 2. FP from bloom filter
                pass
        return (False, None)

    def scan(self, start_key: str, end_key: str) -> Iterator[Tuple[str, Optional[str]]]:
        if not self.tables:
            return
        
        iterators = []
        for i, table in enumerate(reversed(self.tables)):
            it = table.scan(start_key, end_key)
            try:
                key, value = next(it)
                heapq.heappush(iterators, (key, i, value, it))
            except StopIteration:
                pass
        
        last_key = None
        while iterators:
            key, priority, value, it = heapq.heappop(iterators)
            
            if key != last_key:
                yield (key, value)
                last_key = key
            try:
                next_key, next_value = next(it)
                heapq.heappush(iterators, (next_key, priority, next_value, it))
            except StopIteration:
                pass

    def get_all_data(self) -> Iterator[Tuple[str, Optional[str]]]:
        # DONE: написал k-way merge, чтобы не выгружать всё в RAM
        if not self.tables:
            return iter([])
        
        heap = []
        for i, table in enumerate(self.tables):
            it = table.scan("", "\xff" * 100)
            try:
                key, value = next(it)
                heapq.heappush(heap, (key, i, value, it))
            except StopIteration:
                pass
        
        last_key = None
        while heap:
            key, priority, value, it = heapq.heappop(heap)
            
            if key != last_key:
                yield (key, value)
                last_key = key
            
            try:
                next_key, next_value = next(it)
                heapq.heappush(heap, (next_key, priority, next_value, it))
            except StopIteration:
                pass

    def clear(self) -> None:
        for table in self.tables:
            try:
                os.remove(table.filename)
            except OSError as e:
                print(f"Warning: Failed to remove {table.filename}: {e}")
        
        self.tables.clear()

    def __len__(self) -> int:
        return len(self.tables)

    def total_entries(self) -> int:
        return sum(len(table.index) for table in self.tables)