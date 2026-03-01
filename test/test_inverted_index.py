import os
import sys
import shutil
from pyroaring import BitMap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from internal.domain.TextPreprocessor import TextPreprocessor
from internal.domain.InvertedIndex import InvertedIndex
 
 
LOREM_IPSUM_SENTENCES = [
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
    "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.",
    "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
]
 
 
def cleanup_test_data():
    dirs = ["./inverted_index_data", "./test_index_data"]
    for d in dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
 
 
def test_search_single_term():
    index = InvertedIndex(
        preprocessor=TextPreprocessor(language='english'),
        use_lsm=False
    )
 
    for doc in LOREM_IPSUM_SENTENCES:
        index.add_document(doc)
 
    results = index.search("dolor")
    expected = BitMap([0, 2])
    assert results == expected
 
 
def test_search_and():
    index = InvertedIndex(
        preprocessor=TextPreprocessor(language='english'),
        use_lsm=False
    )
 
    for doc in LOREM_IPSUM_SENTENCES:
        index.add_document(doc)
 
    results = index.search_and(["dolor", "sit"])
    expected = BitMap([0])
    assert results == expected
 
 
def test_search_or():
    index = InvertedIndex(
        preprocessor=TextPreprocessor(language='english'),
        use_lsm=False
    )
 
    for doc in LOREM_IPSUM_SENTENCES:
        index.add_document(doc)
 
    results = index.search_or(["lorem", "excepteur"])
    expected = BitMap([0, 3])
    assert results == expected
 
 
def test_search_not():
    index = InvertedIndex(
        preprocessor=TextPreprocessor(language='english'),
        use_lsm=False
    )
 
    for doc in LOREM_IPSUM_SENTENCES:
        index.add_document(doc)
 
    results = index.search_not("dolor")
    expected = BitMap([1, 3])
    assert results == expected
 
 
def test_boolean_query_and():
    index = InvertedIndex(
        preprocessor=TextPreprocessor(language='english'),
        use_lsm=False
    )
 
    for doc in LOREM_IPSUM_SENTENCES:
        index.add_document(doc)
 
    results = index.search_boolean("dolor AND sit")
    expected = BitMap([0])
    assert results == expected
 
 
def test_boolean_query_or():
    index = InvertedIndex(
        preprocessor=TextPreprocessor(language='english'),
        use_lsm=False
    )
 
    for doc in LOREM_IPSUM_SENTENCES:
        index.add_document(doc)
 
    results = index.search_boolean("lorem OR excepteur")
    expected = BitMap([0, 3])
    assert results == expected
 
 
def test_boolean_query_not():
    index = InvertedIndex(
        preprocessor=TextPreprocessor(language='english'),
        use_lsm=False
    )
 
    for doc in LOREM_IPSUM_SENTENCES:
        index.add_document(doc)
 
    results = index.search_boolean("NOT dolor")
    expected = BitMap([1, 3])
    assert results == expected
 
 
def test_boolean_query_complex():
    index = InvertedIndex(
        preprocessor=TextPreprocessor(language='english'),
        use_lsm=False
    )
 
    for doc in LOREM_IPSUM_SENTENCES:
        index.add_document(doc)
 
    results = index.search_boolean("(lorem OR excepteur) AND dolor")
    expected = BitMap([0])
    assert results == expected
 
 
def test_lsm_persistence():
    if os.path.exists("./test_index_data"):
        shutil.rmtree("./test_index_data")
 
    index = InvertedIndex(
        preprocessor=TextPreprocessor(language='english'),
        use_lsm=True,
        lsm_data_dir="./test_index_data"
    )
 
    for doc in LOREM_IPSUM_SENTENCES:
        index.add_document(doc)
 
    index.flush()
    results_before = index.search("dolor")
    index.close()
 
    index2 = InvertedIndex(
        preprocessor=TextPreprocessor(language='english'),
        use_lsm=True,
        lsm_data_dir="./test_index_data"
    )
 
    results_after = index2.search("dolor")
    index2.close()
 
    assert results_before == results_after
 
 
def test_text_preprocessing_tokenization():
    preprocessor = TextPreprocessor(language='english')
    test_text = "Lorem ipsum dolor sit amet"
    tokens = preprocessor.tokenize(test_text)
    expected = ['lorem', 'ipsum', 'dolor', 'sit', 'amet']
    assert tokens == expected
 
 
def test_text_preprocessing_lemmatization():
    preprocessor = TextPreprocessor(language='english')
    test_text = "Lorem ipsum dolor sit amet"
    tokens = preprocessor.tokenize(test_text)
    lemmas = [preprocessor.lemmatize(token) for token in tokens]
    expected = ['lorem', 'ipsum', 'dolor', 'sit', 'amet']
    assert lemmas == expected
 
 
def test_document_retrieval():
    index = InvertedIndex(
        preprocessor=TextPreprocessor(language='english'),
        use_lsm=False
    )
 
    for i, doc in enumerate(LOREM_IPSUM_SENTENCES):
        index.add_document(doc)
 
    retrieved = index.get_document(0)
    assert retrieved == LOREM_IPSUM_SENTENCES[0]
 
 
def test_multiple_documents_retrieval():
    index = InvertedIndex(
        preprocessor=TextPreprocessor(language='english'),
        use_lsm=False
    )
 
    for doc in LOREM_IPSUM_SENTENCES:
        index.add_document(doc)
 
    results = index.search("dolor")
    docs = index.get_documents(results)
    expected = {0: LOREM_IPSUM_SENTENCES[0], 2: LOREM_IPSUM_SENTENCES[2]}
    assert docs == expected
 
 
def test_index_statistics():
    index = InvertedIndex(
        preprocessor=TextPreprocessor(language='english'),
        use_lsm=False
    )
 
    for doc in LOREM_IPSUM_SENTENCES:
        index.add_document(doc)
 
    stats = index.stats()
    assert stats['total_documents'] == len(LOREM_IPSUM_SENTENCES)
 
 
def main():
    try:
        cleanup_test_data()
        test_search_single_term()
        test_search_and()
        test_search_or()
        test_search_not()
        test_boolean_query_and()
        test_boolean_query_or()
        test_boolean_query_not()
        test_boolean_query_complex()
        test_lsm_persistence()
        test_text_preprocessing_tokenization()
        test_text_preprocessing_lemmatization()
        test_document_retrieval()
        test_multiple_documents_retrieval()
        test_index_statistics()
        print("\n" + "="*50)
        print("All tests passed!")
        print("="*50)
 
    except AssertionError as e:
        raise RuntimeError(f"\nTest failed: {e}")
 
    finally:
        cleanup_test_data()
 
 
if __name__ == "__main__":
    main()