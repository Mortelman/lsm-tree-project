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
    print("\n=== Test: K-gram Generation ===")
    index = InvertedIndex(
        preprocessor=TextPreprocessor(language='english'),
        use_lsm=False
    )
    
    kgrams = index._generate_kgrams("hello", k=2)
    expected = ["$h", "he", "el", "ll", "lo", "o$"]
    assert kgrams == expected, f"Expected {expected}, got {kgrams}"
    print(f"✓ Bigrams for 'hello': {kgrams}")
    
    kgrams_short = index._generate_kgrams("hi", k=2)
    expected_short = ["$h", "hi", "i$"]
    assert kgrams_short == expected_short, f"Expected {expected_short}, got {kgrams_short}"
    print(f"✓ Bigrams for 'hi': {kgrams_short}")
    
    kgrams_very_short = index._generate_kgrams("a", k=2)
    expected_very_short = ["$a$"]
    assert kgrams_very_short == expected_very_short, f"Expected {expected_very_short}, got {kgrams_very_short}"
    print(f"✓ Bigrams for 'a': {kgrams_very_short}")
    
    print("✓ K-gram generation test passed")


def test_prefix_search_simple():
    print("\n=== Test: Simple Prefix Search ===")
    index = InvertedIndex(
        preprocessor=TextPreprocessor(language='english'),
        use_lsm=False
    )
    
    for doc in TEST_DOCUMENTS:
        index.add_document(doc)
    
    results = index.search_prefix("hel")
    expected = BitMap([0, 1, 2, 7])
    print(f"Documents with prefix 'hel': {list(results)}")
    print(f"Expected: {list(expected)}")
    
    assert results == expected, f"Expected {list(expected)}, got {list(results)}"
    
    print("✓ Simple prefix search test passed")


def test_prefix_search_not_found():
    print("\n=== Test: Prefix Search Not Found ===")
    index = InvertedIndex(
        preprocessor=TextPreprocessor(language='english'),
        use_lsm=False
    )
    
    for doc in TEST_DOCUMENTS:
        index.add_document(doc)
    
    results = index.search_prefix("xyz")
    assert len(results) == 0, "Should find no documents for non-existent prefix"
    print("✓ Prefix search not found test passed")


def test_prefix_search_with_lsm():
    print("\n=== Test: Prefix Search with LSM ===")
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
    print(f"Documents with prefix 'wor': {list(results)}")
    print(f"Expected: {list(expected)}")
    
    assert results == expected, f"Expected {list(expected)}, got {list(results)}"
    
    index.close()
    print("✓ Prefix search with LSM test passed")


def test_wildcard_suffix():
    print("\n=== Test: Wildcard Suffix (hel*) ===")
    index = InvertedIndex(
        preprocessor=TextPreprocessor(language='english'),
        use_lsm=False
    )
    
    for doc in TEST_DOCUMENTS:
        index.add_document(doc)
    
    results = index.search_wildcard("hel*")
    expected = BitMap([0, 1, 2, 7])
    print(f"Documents matching 'hel*': {list(results)}")
    print(f"Expected: {list(expected)}")
    
    assert results == expected, f"Expected {list(expected)}, got {list(results)}"
    
    print("✓ Wildcard suffix test passed")


def test_wildcard_prefix():
    print("\n=== Test: Wildcard Prefix (*orld) ===")
    index = InvertedIndex(
        preprocessor=TextPreprocessor(language='english'),
        use_lsm=False
    )
    
    for doc in TEST_DOCUMENTS:
        index.add_document(doc)
    
    results = index.search_wildcard("*orld")
    expected = BitMap([0, 3, 7])
    print(f"Documents matching '*orld': {list(results)}")
    print(f"Expected: {list(expected)}")
    
    assert results == expected, f"Expected {list(expected)}, got {list(results)}"
    
    print("✓ Wildcard prefix test passed")


def test_wildcard_middle():
    print("\n=== Test: Wildcard Middle (h*lo) ===")
    index = InvertedIndex(
        preprocessor=TextPreprocessor(language='english'),
        use_lsm=False
    )
    
    for doc in TEST_DOCUMENTS:
        index.add_document(doc)
    
    results = index.search_wildcard("h*lo")
    expected = BitMap([0, 7])
    print(f"Documents matching 'h*lo': {list(results)}")
    print(f"Expected: {list(expected)}")
    
    assert results == expected, f"Expected {list(expected)}, got {list(results)}"
    
    print("✓ Wildcard middle test passed")


def test_wildcard_multiple():
    print("\n=== Test: Multiple Wildcards (p*o*ing) ===")
    index = InvertedIndex(
        preprocessor=TextPreprocessor(language='english'),
        use_lsm=False
    )
    
    for doc in TEST_DOCUMENTS:
        index.add_document(doc)
    
    results = index.search_wildcard("p*o*ing")
    expected = BitMap([5, 6])
    print(f"Documents matching 'p*o*ing': {list(results)}")
    print(f"Expected: {list(expected)}")
    
    assert results == expected, f"Expected {list(expected)}, got {list(results)}"
    
    print("✓ Multiple wildcards test passed")


def test_wildcard_only_star():
    print("\n=== Test: Wildcard Only Star (*) ===")
    index = InvertedIndex(
        preprocessor=TextPreprocessor(language='english'),
        use_lsm=False
    )
    
    for doc in TEST_DOCUMENTS:
        index.add_document(doc)
    
    results = index.search_wildcard("*")
    expected = BitMap(range(len(TEST_DOCUMENTS)))
    print(f"Documents matching '*': {list(results)}")
    print(f"Expected: {list(expected)}")
    
    assert results == expected, f"Expected {list(expected)}, got {list(results)}"
    
    print("✓ Wildcard only star test passed")


def test_wildcard_not_found():
    print("\n=== Test: Wildcard Not Found ===")
    index = InvertedIndex(
        preprocessor=TextPreprocessor(language='english'),
        use_lsm=False
    )
    
    for doc in TEST_DOCUMENTS:
        index.add_document(doc)
    
    results = index.search_wildcard("xyz*abc")
    assert len(results) == 0, "Should find no documents"
    
    print("✓ Wildcard not found test passed")


def test_wildcard_with_lsm():
    print("\n=== Test: Wildcard with LSM ===")
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
    print(f"Documents matching 'pro*ing': {list(results)}")
    
    assert 5 in results, "Document 5 (programming) should be found"
    assert 6 in results, "Document 6 (programming) should be found"
    
    index.close()
    print("✓ Wildcard with LSM test passed")


def test_kgram_index_building():
    print("\n=== Test: K-gram Index Building ===")
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
    
    print(f"✓ K-gram index contains {len(index.kgram_index)} k-grams")
    print("✓ K-gram index building test passed")


def main():
    try:
        cleanup_test_data()
        
        print("="*60)
        print("Running Prefix and Wildcard Search Tests")
        print("="*60)
        
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
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
    
    finally:
        cleanup_test_data()


if __name__ == "__main__":
    main()