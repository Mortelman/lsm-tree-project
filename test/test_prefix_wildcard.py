import os
import sys
import shutil
from pyroaring import BitMap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from internal.domain.TextPreprocessor import TextPreprocessor
from internal.domain.InvertedIndex import InvertedIndex


TEST_DOCUMENTS = [
    "Hello world, this is a test document",
    "Help me with this problem please",
    "Helicopter flying over the city",
    "World peace is important for everyone",
    "The quick brown fox jumps over the lazy dog",
    "Python programming is fun and powerful",
    "Programming languages include Python, Java, and C++",
    "Hello again, welcome back to the world"
]


def cleanup_test_data():
    dirs = ["./test_prefix_wildcard_data", "./test_prefix_wildcard_lsm"]
    for d in dirs:
        if os.path.exists(d):
            shutil.rmtree(d)


def test_kgram_generation():
    index = InvertedIndex(
        preprocessor=TextPreprocessor(language='english'),
        use_lsm=False
    )
    
    kgrams = index._generate_kgrams("hello", k=2)
    expected = ["$h", "he", "el", "ll", "lo", "o$"]
    assert kgrams == expected, f"Expected {expected}, got {kgrams}"
    
    kgrams_short = index._generate_kgrams("hi", k=2)
    expected_short = ["$h", "hi", "i$"]
    assert kgrams_short == expected_short, f"Expected {expected_short}, got {kgrams_short}"
    
    kgrams_very_short = index._generate_kgrams("a", k=2)
    expected_very_short = ["$a$"]
    assert kgrams_very_short == expected_very_short, f"Expected {expected_very_short}, got {kgrams_very_short}"


def test_prefix_search_simple():
    index = InvertedIndex(
        preprocessor=TextPreprocessor(language='english'),
        use_lsm=False
    )
    
    for doc in TEST_DOCUMENTS:
        index.add_document(doc)
    
    results = index.search_prefix("hel")
    expected = BitMap([0, 1, 2, 7])
    
    assert results == expected, f"Expected {list(expected)}, got {list(results)}"


def test_prefix_search_not_found():
    index = InvertedIndex(
        preprocessor=TextPreprocessor(language='english'),
        use_lsm=False
    )
    
    for doc in TEST_DOCUMENTS:
        index.add_document(doc)
    
    results = index.search_prefix("xyz")
    assert len(results) == 0, "Should find no documents for non-existent prefix"


def test_prefix_search_with_lsm():
    cleanup_test_data()
    
    index = InvertedIndex(
        preprocessor=TextPreprocessor(language='english'),
        use_lsm=True,
        lsm_data_dir="./test_prefix_wildcard_lsm"
    )
    
    for doc in TEST_DOCUMENTS:
        index.add_document(doc)
    
    index.flush()
    
    results = index.search_prefix("wor")
    expected = BitMap([0, 3, 7])
    
    assert results == expected, f"Expected {list(expected)}, got {list(results)}"
    index.close()


def test_wildcard_suffix():
    index = InvertedIndex(
        preprocessor=TextPreprocessor(language='english'),
        use_lsm=False
    )
    
    for doc in TEST_DOCUMENTS:
        index.add_document(doc)
    
    results = index.search_wildcard("hel*")
    expected = BitMap([0, 1, 2, 7])
    assert results == expected, f"Expected {list(expected)}, got {list(results)}"


def test_wildcard_prefix():
    index = InvertedIndex(
        preprocessor=TextPreprocessor(language='english'),
        use_lsm=False
    )
    
    for doc in TEST_DOCUMENTS:
        index.add_document(doc)
    
    results = index.search_wildcard("*orld")
    expected = BitMap([0, 3, 7])
    assert results == expected, f"Expected {list(expected)}, got {list(results)}"

def test_wildcard_middle():
    index = InvertedIndex(
        preprocessor=TextPreprocessor(language='english'),
        use_lsm=False
    )
    
    for doc in TEST_DOCUMENTS:
        index.add_document(doc)
    
    results = index.search_wildcard("h*lo")
    expected = BitMap([0, 7])
    assert results == expected, f"Expected {list(expected)}, got {list(results)}"

def test_wildcard_multiple():
    index = InvertedIndex(
        preprocessor=TextPreprocessor(language='english'),
        use_lsm=False
    )
    
    for doc in TEST_DOCUMENTS:
        index.add_document(doc)
    
    results = index.search_wildcard("p*o*ing")
    expected = BitMap([5, 6])
    assert results == expected, f"Expected {list(expected)}, got {list(results)}"


def test_wildcard_only_star():
    index = InvertedIndex(
        preprocessor=TextPreprocessor(language='english'),
        use_lsm=False
    )
    
    for doc in TEST_DOCUMENTS:
        index.add_document(doc)
    
    results = index.search_wildcard("*")
    expected = BitMap(range(len(TEST_DOCUMENTS)))  
    assert results == expected, f"Expected {list(expected)}, got {list(results)}"


def test_wildcard_not_found():
    index = InvertedIndex(
        preprocessor=TextPreprocessor(language='english'),
        use_lsm=False
    )
    
    for doc in TEST_DOCUMENTS:
        index.add_document(doc)
    
    results = index.search_wildcard("xyz*abc")
    assert len(results) == 0, "Should find no documents"


def test_wildcard_with_lsm():
    cleanup_test_data()
    
    index = InvertedIndex(
        preprocessor=TextPreprocessor(language='english'),
        use_lsm=True,
        lsm_data_dir="./test_prefix_wildcard_lsm"
    )
    
    for doc in TEST_DOCUMENTS:
        index.add_document(doc)
    
    index.flush()
    
    results = index.search_wildcard("pro*ing")
    
    assert 5 in results, "Document 5 (programming) should be found"
    assert 6 in results, "Document 6 (programming) should be found"   
    index.close()


def test_kgram_index_building():
    index = InvertedIndex(
        preprocessor=TextPreprocessor(language='english'),
        use_lsm=False
    )
    
    index.add_document("Hello world")
    
    assert len(index.kgram_index) > 0, "K-gram index should not be empty"
    hello_kgrams = index._generate_kgrams("hello", k=2)
    for kgram in hello_kgrams:
        assert kgram in index.kgram_index, f"K-gram '{kgram}' should be in index"
        assert "hello" in index.kgram_index[kgram], f"'hello' should be in k-gram '{kgram}'"

def main():
    try:
        cleanup_test_data()
        
        print("="*50)
        print("Running Prefix and Wildcard Search Tests")
        print("="*50)
        
        test_kgram_generation()
        test_kgram_index_building()
        
        test_prefix_search_simple()
        test_prefix_search_not_found()
        test_prefix_search_with_lsm()
        
        test_wildcard_suffix()
        test_wildcard_prefix()
        test_wildcard_middle()
        test_wildcard_multiple()
        test_wildcard_only_star()
        test_wildcard_not_found()
        test_wildcard_with_lsm()
        
        print("\n" + "="*50)
        print(" ALL TESTS PASSED!")
        print("="*50)
        
    except AssertionError as e:
        print(f"\n TEST FAILED: {e}")
        raise
    
    finally:
        cleanup_test_data()


if __name__ == "__main__":
    main()