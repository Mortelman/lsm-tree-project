from typing import Optional, List, Tuple, Iterator, Dict
from sortedcontainers import SortedDict


class MemTable:

    def __init__(self, max_size: int = 1024 * 1024):
        self._data: SortedDict = SortedDict()
        self._size: int = 0
        self._max_size: int = max_size

    def put(self, key: str, value: str) -> None:
        old_size = self._get_entry_size(key, self._data.get(key))
        new_size = self._get_entry_size(key, value)
        
        self._data[key] = value
        self._size += (new_size - old_size)

    def get(self, key: str) -> Tuple[bool, Optional[str]]:
        if key in self._data:
            return (True, self._data[key])
        return (False, None)

    def delete(self, key: str) -> None:
        old_size = self._get_entry_size(key, self._data.get(key))
        new_size = self._get_entry_size(key, None)
        
        self._data[key] = None
        self._size += (new_size - old_size)

    def scan(self, start_key: str, end_key: str) -> Iterator[Tuple[str, Optional[str]]]:
        for key in self._data.irange(start_key, end_key, inclusive=(True, True)):
            yield (key, self._data[key])

    def is_full(self) -> bool:
        return self._size >= self._max_size

    def size(self) -> int:
        return self._size

    def __len__(self) -> int:
        return len(self._data)

    def items(self) -> List[Tuple[str, Optional[str]]]:
        return list(self._data.items())

    def clear(self) -> None:
        self._data.clear()
        self._size = 0

    def _get_entry_size(self, key: str, value: Optional[str]) -> int:
        if key not in self._data and value is None:
            return 0
        
        size = len(key.encode('utf-8'))
        if value is not None:
            size += len(value.encode('utf-8'))
        else:
            size += 1
        return size