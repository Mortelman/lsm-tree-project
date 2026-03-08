import pickle
import re
from typing import List, Dict, Set, Optional, Tuple
from pyroaring import BitMap
from .TextPreprocessor import TextPreprocessor
from .LSMTree import LSMTree


class InvertedIndex:
    
    def __init__(
        self,
        preprocessor: Optional[TextPreprocessor] = None,
        use_lsm: bool = True,
        lsm_data_dir: str = "./inverted_index_data"
    ):
        self.preprocessor = preprocessor or TextPreprocessor()
        self.use_lsm = use_lsm
        
        self.index: Dict[str, BitMap] = {}
        self.documents: Dict[int, str] = {}
        self.next_doc_id: int = 0
        
        self.kgram_index: Dict[str, Set[str]] = {}
        self.k: int = 2
        
        if use_lsm:
            self.lsm = LSMTree(
                data_dir=lsm_data_dir,
                memtable_size=1024 * 1024,
                max_levels=5,
                level_ratio=10
            )
        else:
            self.lsm = None
    
    def add_document(self, text: str, doc_id: Optional[int] = None) -> int:
        if doc_id is None:
            doc_id = self.next_doc_id
            self.next_doc_id += 1
        else:
            self.next_doc_id = max(self.next_doc_id, doc_id + 1)
        
        self.documents[doc_id] = text

        tokens = self.preprocessor.preprocess(text)
        
        for token in tokens:
            if token not in self.index:
                self.index[token] = BitMap()
                self._add_to_kgram_index(token)
            self.index[token].add(doc_id)
        
        if self.use_lsm:
            self._persist_document(doc_id, text, tokens)
            for token in set(tokens):
                if token in self.index:
                    self._persist_kgrams(token)
        
        return doc_id
    
    def _persist_document(self, doc_id: int, text: str, tokens: List[str]) -> None:
        doc_key = f"doc:{doc_id}"
        self.lsm.put(doc_key, text)
        
        for token in set(tokens):
            index_key = f"idx:{token}"
            
            existing = self.lsm.get(index_key)
            if existing:
                bitmap = BitMap.deserialize(existing.encode('latin1'))
            else:
                bitmap = BitMap()
            
            bitmap.add(doc_id)
            
            serialized = bitmap.serialize().decode('latin1')
            self.lsm.put(index_key, serialized)
    
    def search(self, term: str) -> BitMap:
        tokens = self.preprocessor.preprocess(term)
        
        if not tokens:
            return BitMap()
        
        token = tokens[0]
        if token in self.index:
            return self.index[token].copy()
        
        if self.use_lsm:
            index_key = f"idx:{token}"
            serialized = self.lsm.get(index_key)
            if serialized:
                return BitMap.deserialize(serialized.encode('latin1'))
        
        return BitMap()
    
    def search_and(self, terms: List[str]) -> BitMap:
        if not terms:
            return BitMap()
        
        result = self.search(terms[0])
        for term in terms[1:]:
            result &= self.search(term)
        
        return result
    
    def search_or(self, terms: List[str]) -> BitMap:
        result = BitMap()
        
        for term in terms:
            result |= self.search(term)
        
        return result
    
    def search_not(self, term: str) -> BitMap:
        all_docs = BitMap(range(self.next_doc_id))
        term_docs = self.search(term)
        return all_docs - term_docs
    
    def search_boolean(self, query: str) -> BitMap:
        return self._parse_boolean_query(query)
    
    def _parse_boolean_query(self, query: str) -> BitMap:
        query = query.strip()
        
        if query.startswith('(') and query.endswith(')'):
            return self._parse_boolean_query(query[1:-1])
        
        if query.upper().startswith('NOT '):
            term = query[4:].strip()
            return self.search_not(term)
        
        if ' AND ' in query.upper():
            parts = query.upper().split(' AND ')
            and_pos = query.upper().index(' AND ')
            left = query[:and_pos].strip()
            right = query[and_pos + 5:].strip()
            return self._parse_boolean_query(left) & self._parse_boolean_query(right)
        
        if ' OR ' in query.upper():
            parts = query.upper().split(' OR ')
            or_pos = query.upper().index(' OR ')
            left = query[:or_pos].strip()
            right = query[or_pos + 4:].strip()
            return self._parse_boolean_query(left) | self._parse_boolean_query(right)
        
        return self.search(query)
    
    def get_document(self, doc_id: int) -> Optional[str]:

        if doc_id in self.documents:
            return self.documents[doc_id]
        
        if self.use_lsm:
            doc_key = f"doc:{doc_id}"
            return self.lsm.get(doc_key)
        
        return None
    
    def get_documents(self, doc_ids: BitMap) -> Dict[int, str]:
        result = {}
        for doc_id in doc_ids:
            doc = self.get_document(doc_id)
            if doc:
                result[doc_id] = doc
        return result
    
    def flush(self) -> None:
        if self.use_lsm:
            self.lsm.flush()
    
    def close(self) -> None:
        self.flush()
        if self.use_lsm:
            self.lsm.close()
    
    def stats(self) -> Dict:
        stats = {
            "total_documents": self.next_doc_id,
            "total_terms": len(self.index),
            "in_memory_documents": len(self.documents),
        }
        
        if self.use_lsm:
            stats["lsm_stats"] = self.lsm.stats()
        
        return stats
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    def _generate_kgrams(self, term: str, k: int = 2) -> List[str]:
        if len(term) < k:
            return [f"${term}$"]
        
        padded = f"${term}$"
        return [padded[i:i+k] for i in range(len(padded) - k + 1)]
    
    def _add_to_kgram_index(self, term: str) -> None:
        kgrams = self._generate_kgrams(term, self.k)
        for kgram in kgrams:
            if kgram not in self.kgram_index:
                self.kgram_index[kgram] = set()
            self.kgram_index[kgram].add(term)
    
    def _persist_kgrams(self, term: str) -> None:
        if not self.use_lsm:
            return
        
        kgrams = self._generate_kgrams(term, self.k)
        for kgram in kgrams:
            kgram_key = f"kgram:{kgram}"
            existing = self.lsm.get(kgram_key)
            if existing:
                terms = set(existing.split(','))
            else:
                terms = set()
            
            terms.add(term)
            self.lsm.put(kgram_key, ','.join(sorted(terms)))
    
    def _generate_kgrams_for_wildcard_part(self, part: str, pattern: str, part_index: int, total_parts: int) -> List[str]:
        starts_pattern = pattern.startswith(part)
        ends_pattern = pattern.endswith(part)
        
        kgrams = []
        
        if len(part) < self.k:
            if starts_pattern:
                kgrams.append(f"${part}")
            if ends_pattern:
                kgrams.append(f"{part}$")
            if not kgrams:
                return []
            return kgrams
        
        if starts_pattern:
            padded = f"${part}"
        else:
            padded = part
        for i in range(len(padded) - self.k + 1):
            kgrams.append(padded[i:i+self.k])
        
        if ends_pattern and len(part) >= self.k:
            kgrams.append(part[-(self.k-1):] + "$")
        
        return kgrams
    
    def _find_wildcard_candidates(self, parts: List[str], pattern: str) -> Set[str]:
        candidate_terms = None
        
        for i, part in enumerate(parts):
            kgrams = self._generate_kgrams_for_wildcard_part(part, pattern, i, len(parts))
            if not kgrams:
                all_terms = set(self.index.keys())
                if self.use_lsm:
                    pass
                part_candidates = all_terms
            else:
                part_candidates = None
                for kgram in kgrams:
                    terms = self.kgram_index.get(kgram, set()).copy()
                    if self.use_lsm:
                        kgram_key = f"kgram:{kgram}"
                        existing = self.lsm.get(kgram_key)
                        if existing:
                            terms.update(existing.split(','))
                    
                    if part_candidates is None:
                        part_candidates = terms
                    else:
                        part_candidates &= terms
                    if not part_candidates:
                        return set()
            
            if candidate_terms is None:
                candidate_terms = part_candidates if part_candidates else set()
            else:
                if part_candidates:
                    candidate_terms &= part_candidates
                else:
                    return set()
            if not candidate_terms:
                return set()
        
        return candidate_terms or set()
    
    def search_prefix(self, prefix: str) -> BitMap:
        processed_prefix = self.preprocessor.lemmatize(prefix.lower())
        result = BitMap()
        for term in self.index.keys():
            if term.startswith(processed_prefix):
                result |= self.index[term]
        if self.use_lsm:
            start_key = f"idx:{processed_prefix}"
            end_key = f"idx:{processed_prefix}\xff"
            for key, serialized_bitmap in self.lsm.scan(start_key, end_key):
                if key.startswith("idx:"):
                    term = key[4:]
                    if term.startswith(processed_prefix):
                        bitmap = BitMap.deserialize(serialized_bitmap.encode('latin1'))
                        result |= bitmap
        
        return result
    
    def search_wildcard(self, pattern: str) -> BitMap:
        pattern = pattern.lower()
        parts = [p for p in pattern.split('*') if p]
        if not parts:
            return BitMap(range(self.next_doc_id))
        candidate_terms = self._find_wildcard_candidates(parts, pattern)
        if not candidate_terms:
            return BitMap()
        
        regex_pattern = '^' + pattern.replace('*', '.*') + '$'
        regex = re.compile(regex_pattern)
        
        matching_terms = [t for t in candidate_terms if regex.match(t)]
        result = BitMap()
        for term in matching_terms:
            if term in self.index:
                result |= self.index[term]
            elif self.use_lsm:
                index_key = f"idx:{term}"
                serialized = self.lsm.get(index_key)
                if serialized:
                    result |= BitMap.deserialize(serialized.encode('latin1'))
        
        return result