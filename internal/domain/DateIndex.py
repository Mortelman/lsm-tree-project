from datetime import datetime
from typing import Optional
from pyroaring import BitMap
from .BitSlicedIndex import BitSlicedIndex


class DateIndex:
    
    def __init__(self):
        self.index = BitSlicedIndex(num_bits=64)
        self.null_bitmap = BitMap()
    
    def add(self, doc_id: int, date: Optional[datetime]) -> None:
        if date is None:
            self.null_bitmap.add(doc_id)
        else:
            timestamp = int(date.timestamp())
            self.index.add(doc_id, timestamp)
    
    def search_range(self, start: datetime, end: datetime, inclusive: bool = True) -> BitMap:
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())
        return self.index.range(start_ts, end_ts, inclusive=inclusive)
    
    def search_before(self, date: datetime, inclusive: bool = False) -> BitMap:
        timestamp = int(date.timestamp())
        if inclusive:
            return self.index.less_equal(timestamp)
        else:
            return self.index.less_than(timestamp)
    
    def search_after(self, date: datetime, inclusive: bool = False) -> BitMap:
        timestamp = int(date.timestamp())
        if inclusive:
            return self.index.greater_equal(timestamp)
        else:
            return self.index.greater_than(timestamp)
    
    def search_equals(self, date: datetime) -> BitMap:
        timestamp = int(date.timestamp())
        return self.index.equals(timestamp)
    
    def search_null(self) -> BitMap:
        return self.null_bitmap.copy()
    
    def search_not_null(self) -> BitMap:
        all_docs = BitMap()
        for slice_bm in self.index.bit_slices:
            all_docs |= slice_bm
        return all_docs