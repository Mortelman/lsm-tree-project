import hashlib
import pickle

class BloomFilter:

    def __init__(self, m: int = 100, k: int = 10, seed: int = 137):
        self.m = m
        self.k = k
        self.seed = seed
        self.bits = 0

    def _hashes(self, key: str) -> list[int]:
        hashes = [] 
        key_bytes = key.encode()
        for i in range(self.k):
            h = hashlib.md5(key_bytes + str(self.seed * (i + 1)).encode()).digest()
            num = int.from_bytes(h)
            hashes.append(num % self.m)
        return hashes
    
    def add(self, key: str) -> None:
        for pos in self._hashes(key):
            self.bits |= (1 << pos)

    def __contains__(self, key: str) -> bool:
        for pos in self._hashes(key):
            if not (self.bits & (1 << pos)):
                return False
        return True
    
    def serialize(self) -> bytes:
        return pickle.dumps((self.m, self.k, self.seed, self.bits))
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'BloomFilter':
        m, k, seed, bits = pickle.loads(data)
        bf = cls(m=m, k=k, seed=seed)
        bf.bits = bits
        return bf
    

