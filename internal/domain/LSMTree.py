import os
import heapq
from typing import Optional, List, Tuple, Iterator
 
from .MemTable import MemTable
from .Level import Level
 
 
class LSMTree:
 
    def __init__(
        self,
        data_dir: str = "./lsm_data",
        memtable_size: int = 1024 * 1024,
        max_levels: int = 5,
        level_ratio: int = 10,
    ):
        self.data_dir = data_dir
        self.memtable_size = memtable_size
        self.max_levels = max_levels
        self.level_ratio = level_ratio
 
        os.makedirs(data_dir, exist_ok=True)
 
        self._memtable = MemTable(max_size=memtable_size)
 
        self._levels: List[Level] = []
        for i in range(max_levels):
            level = Level(i, data_dir, max_tables=level_ratio)
            self._levels.append(level)
 
    def put(self, key: str, value: str) -> None:
        self._memtable.put(key, value)
 
        if self._memtable.is_full():
            self._flush_memtable()
 
    def get(self, key: str) -> Optional[str]:
        found, value = self._memtable.get(key)
        if found:
            return value
 
        for level in self._levels:
            found, value = level.get(key)
            if found:
                return value
 
        return None
 
    def delete(self, key: str) -> None:
        self._memtable.delete(key)
 
        if self._memtable.is_full():
            self._flush_memtable()
 
    def scan(self, start_key: str, end_key: str) -> Iterator[Tuple[str, str]]:
        sources = []
        sources.append((0, self._memtable.scan(start_key, end_key)))
 
        for i, level in enumerate(self._levels):
            sources.append((i + 1, level.scan(start_key, end_key)))
 
        heap = []
        for priority, it in sources:
            try:
                key, value = next(it)
                heapq.heappush(heap, (key, priority, value, it))
            except StopIteration:
                pass
 
        last_key = None
        while heap:
            key, priority, value, it = heapq.heappop(heap)
 
            if key != last_key:
                if value is not None:
                    yield (key, value)
                last_key = key
 
            try:
                next_key, next_value = next(it)
                heapq.heappush(heap, (next_key, priority, next_value, it))
            except StopIteration:
                pass
 
    def _flush_memtable(self) -> None:
        if len(self._memtable) == 0:
            return
 
        data = self._memtable.items()
        self._memtable = MemTable(max_size=self.memtable_size)
 
        self._levels[0].add_table(data)
 
        self._maybe_compact()
 
    def _maybe_compact(self) -> None:
        for i in range(len(self._levels) - 1):
            level = self._levels[i]
 
            if level.needs_compaction():
                self._compact_level(i)
 
    def _compact_level(self, level_num: int) -> None:
        if level_num >= len(self._levels) - 1:
            return
 
        current_level = self._levels[level_num]
        next_level = self._levels[level_num + 1]
 
        merged_data = current_level.get_all_data()
 
        if merged_data:
            next_level.add_table(merged_data)
 
        current_level.clear()
 
        if next_level.needs_compaction():
            self._compact_level(level_num + 1)
 
    def flush(self) -> None:
        self._flush_memtable()
 
    def close(self) -> None:
        self.flush()
 
    def stats(self) -> dict:
        level_stats = []
        for i, level in enumerate(self._levels):
            level_stats.append({
                "level": i,
                "tables": len(level),
                "entries": level.total_entries(),
            })
 
        return {
            "memtable_size": self._memtable.size(),
            "memtable_entries": len(self._memtable),
            "levels": level_stats,
        }
 
    def __enter__(self):
        return self
 
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
