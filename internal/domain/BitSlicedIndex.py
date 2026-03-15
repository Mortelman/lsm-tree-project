from typing import List, Optional
from pyroaring import BitMap


class BitSlicedIndex:
    
    def __init__(self, num_bits: int = 64):
        self.num_bits = num_bits
        self.bit_slices: List[BitMap] = [BitMap() for _ in range(num_bits)]
        self.all_docs: BitMap = BitMap()
        self.min_value: Optional[int] = None
        self.max_value: Optional[int] = None
        self.doc_count: int = 0
    
    def add(self, doc_id: int, value: int) -> None:
        if value < 0:
            raise ValueError(f"Value must be non-negative, got {value}")
        
        self.all_docs.add(doc_id)
        
        if self.min_value is None or value < self.min_value:
            self.min_value = value
        if self.max_value is None or value > self.max_value:
            self.max_value = value
        
        for bit_position in range(self.num_bits):
            if (value >> bit_position) & 1:
                self.bit_slices[bit_position].add(doc_id)
        
        self.doc_count += 1
    
    def equals(self, value: int) -> BitMap:
        if self.min_value is None or self.max_value is None:
            return BitMap()
        if value < self.min_value or value > self.max_value:
            return BitMap()
        
        result = None
        
        for bit_position in range(self.num_bits):
            bit_value = (value >> bit_position) & 1
            
            if bit_value == 1:
                current = self.bit_slices[bit_position].copy()
            else:
                all_docs = BitMap()
                for slice_bm in self.bit_slices:
                    all_docs |= slice_bm
                current = all_docs - self.bit_slices[bit_position]
            
            if result is None:
                result = current
            else:
                result &= current
        
        return result if result is not None else BitMap()
    
    def not_equals(self, value: int) -> BitMap:
        all_docs = BitMap()
        for slice_bm in self.bit_slices:
            all_docs |= slice_bm
        
        return all_docs - self.equals(value)
    
    def greater_equal(self, value: int) -> BitMap:
        if self.min_value is None:
            return BitMap()
        
        if value <= self.min_value:
            return self.all_docs.copy()
        
        if self.max_value is not None and value > self.max_value:
            return BitMap()
        
        result = BitMap()
        equal_so_far = self.all_docs.copy()
        
        for bit_position in range(self.num_bits - 1, -1, -1):
            bit_value = (value >> bit_position) & 1
            current_slice = self.bit_slices[bit_position]
            
            if bit_value == 0:
                result |= equal_so_far & current_slice
                equal_so_far &= (self.all_docs - current_slice)
            else:
                equal_so_far &= current_slice
        
        result |= equal_so_far
        
        return result
    
    def greater_than(self, value: int) -> BitMap:
        return self.greater_equal(value + 1)
    
    def less_equal(self, value: int) -> BitMap:
        if self.min_value is None:
            return BitMap()
        
        if self.max_value is not None and value >= self.max_value:
            return self.all_docs.copy()
        
        if value < self.min_value:
            return BitMap()
        
        result = BitMap()
        equal_so_far = self.all_docs.copy()
        
        for bit_position in range(self.num_bits - 1, -1, -1):
            bit_value = (value >> bit_position) & 1
            current_slice = self.bit_slices[bit_position]
            
            if bit_value == 1:
                result |= equal_so_far & (self.all_docs - current_slice)
                equal_so_far &= current_slice
            else:
                equal_so_far &= (self.all_docs - current_slice)
        
        result |= equal_so_far
        
        return result
    
    def less_than(self, value: int) -> BitMap:
        if value == 0:
            return BitMap()
        return self.less_equal(value - 1)
    
    def range(self, start: int, end: int, inclusive: bool = True) -> BitMap:
        if start > end:
            return BitMap()
        
        if inclusive:
            ge_start = self.greater_equal(start)
            le_end = self.less_equal(end)
        else:
            ge_start = self.greater_than(start)
            le_end = self.less_than(end)
        
        return ge_start & le_end