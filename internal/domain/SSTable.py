import pickle
import struct
import bisect
from typing import Optional, List, Tuple, Iterator, Union
from .BloomFilter import BloomFilter


class SSTable:
    MAGIC = b'SSTB'
    VERSION = 1

    def __init__(self, filename: str, data: Optional[Union[List[Tuple[str, Optional[str]]], Iterator[Tuple[str, Optional[str]]]]] = None):
        self.filename = filename
        self.index: List[Tuple[str, int]] = []
        
        if data is not None:
            self._create(data)
        else:
            self._load()
    
    def _create(self, data: Union[List[Tuple[str, Optional[str]]], Iterator[Tuple[str, Optional[str]]]]) -> None:
        if not isinstance(data, list):
            data = list(data)
        
        data.sort(key=lambda x: x[0])

        self.bloom = BloomFilter()
        for key, _ in data:
            self.bloom.add(key)

        with open(self.filename, 'wb') as f:
            f.write(self.MAGIC)
            f.write(struct.pack('>I', len(data)))

            self.index = []
            for key, value in data:
                offset = f.tell()
                self.index.append((key, offset))

                key_b = key.encode('utf-8')
                f.write(struct.pack('>H', len(key_b)))
                f.write(key_b)

                if value is None:
                    f.write(b'\x00')
                else:
                    f.write(b'\x01')
                    value_b = value.encode('utf-8')
                    f.write(struct.pack('>I', len(value_b)))
                    f.write(value_b)

            index_bytes = pickle.dumps(self.index)
            f.write(struct.pack('>I', len(index_bytes)))
            f.write(index_bytes)

            bloom_bytes = self.bloom.serialize()
            f.write(struct.pack('>I', len(bloom_bytes)))
            f.write(bloom_bytes)

    def _load(self) -> None:
        with open(self.filename, 'rb') as f:
            if f.read(4) != self.MAGIC:
                raise ValueError(f"Invalid SSTable format: {self.filename}")

            f.seek(4)
            count = struct.unpack('>I', f.read(4))[0]
            
            for _ in range(count):
                key_len = struct.unpack('>H', f.read(2))[0]
                f.read(key_len)
                flag = f.read(1)[0]
                if flag == 1:
                    value_len = struct.unpack('>I', f.read(4))[0]
                    f.read(value_len)
            
            index_size = struct.unpack('>I', f.read(4))[0]
            index_bytes = f.read(index_size)
            self.index = pickle.loads(index_bytes)
            
            bloom_size = struct.unpack('>I', f.read(4))[0]
            bloom_bytes = f.read(bloom_size)
            self.bloom = BloomFilter.deserialize(bloom_bytes)

    def get(self, key: str) -> Optional[str]:
        if key not in self.bloom:
            return None

        i = bisect.bisect_left(self.index, (key,))
        
        if i >= len(self.index) or self.index[i][0] != key:
            return None

        offset = self.index[i][1]
        
        with open(self.filename, 'rb') as f:
            f.seek(offset)
            
            key_len = struct.unpack('>H', f.read(2))[0]
            f.read(key_len)
            
            flag = f.read(1)[0]
            if flag == 0:
                return None
            
            value_len = struct.unpack('>I', f.read(4))[0]
            value = f.read(value_len).decode('utf-8')
            return value

    def scan(self, start_key: str, end_key: str) -> Iterator[Tuple[str, Optional[str]]]:
        start_idx = bisect.bisect_left(self.index, (start_key,))
        end_idx = bisect.bisect_right(self.index, (end_key + '\xff',))
        
        with open(self.filename, 'rb') as f:
            for i in range(start_idx, end_idx):
                key, offset = self.index[i]
                
                f.seek(offset)
                
                key_len = struct.unpack('>H', f.read(2))[0]
                f.read(key_len)
                
                flag = f.read(1)[0]
                if flag == 0:
                    yield (key, None)
                else:
                    value_len = struct.unpack('>I', f.read(4))[0]
                    value = f.read(value_len).decode('utf-8')
                    yield (key, value)