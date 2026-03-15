import os
import sys
import shutil
from datetime import datetime, timedelta
from pyroaring import BitMap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from internal.domain.TextPreprocessor import TextPreprocessor
from internal.domain.InvertedIndex import InvertedIndex
from internal.domain.BitSlicedIndex import BitSlicedIndex
from internal.domain.DateIndex import DateIndex


def cleanup_test_data():
    dirs = ["./test_date_search_data"]
    for d in dirs:
        if os.path.exists(d):
            shutil.rmtree(d)


def test_bit_sliced_add_and_equals():
    index = BitSlicedIndex(num_bits=8)
    
    index.add(0, 5)
    index.add(1, 3)
    index.add(2, 7)
    index.add(3, 2)
    
    result = index.equals(5)
    assert result == BitMap([0]), f"Expected [0], got {list(result)}"
    
    result = index.equals(3)
    assert result == BitMap([1]), f"Expected [1], got {list(result)}"
    


def test_bit_sliced_greater_equal():
    index = BitSlicedIndex(num_bits=8)
    
    for i in range(10):
        index.add(i, i)
    
    result = index.greater_equal(5)
    expected = BitMap([5, 6, 7, 8, 9])
    assert result == expected, f"Expected {list(expected)}, got {list(result)}"
    
    result = index.greater_equal(0)
    expected = BitMap(range(10))
    assert result == expected, f"Expected {list(expected)}, got {list(result)}"
    

def test_bit_sliced_less_equal():
    index = BitSlicedIndex(num_bits=8)
    
    for i in range(10):
        index.add(i, i)
    
    result = index.less_equal(5)
    expected = BitMap([0, 1, 2, 3, 4, 5])
    assert result == expected, f"Expected {list(expected)}, got {list(result)}"
    
    result = index.less_equal(9)
    expected = BitMap(range(10))
    assert result == expected, f"Expected {list(expected)}, got {list(result)}"


def test_bit_sliced_range():
    index = BitSlicedIndex(num_bits=8)
    
    for i in range(10):
        index.add(i, i)
    
    result = index.range(3, 7)
    expected = BitMap([3, 4, 5, 6, 7])
    assert result == expected, f"Expected {list(expected)}, got {list(result)}"
    
    result = index.range(0, 9)
    expected = BitMap(range(10))
    assert result == expected, f"Expected {list(expected)}, got {list(result)}"


def test_date_index_add_and_search():
    index = DateIndex()
    
    date1 = datetime(2024, 1, 15, 10, 30, 0)
    date2 = datetime(2024, 3, 20, 14, 15, 0)
    date3 = datetime(2024, 6, 10, 9, 0, 0)
    
    index.add(0, date1)
    index.add(1, date2)
    index.add(2, date3)
    
    result = index.search_range(
        datetime(2024, 1, 1),
        datetime(2024, 3, 31)
    )
    expected = BitMap([0, 1])
    assert result == expected, f"Expected {list(expected)}, got {list(result)}"


def test_date_index_null_handling():
    index = DateIndex()
    
    index.add(0, datetime(2024, 1, 15))
    index.add(1, None)
    index.add(2, datetime(2024, 3, 20))
    
    result = index.search_null()
    expected = BitMap([1])
    assert result == expected, f"Expected {list(expected)}, got {list(result)}"
    
    result = index.search_not_null()
    assert 0 in result and 2 in result, f"Expected 0 and 2 in result, got {list(result)}"
    assert 1 not in result, f"Expected 1 not in result, got {list(result)}"


def test_inverted_index_with_created_at():
    index = InvertedIndex(
        preprocessor=TextPreprocessor(language='english'),
        use_lsm=False,
        enable_date_index=True
    )
    
    index.add_document(
        "Python programming tutorial",
        created_at=datetime(2024, 1, 15, 10, 30, 0)
    )
    index.add_document(
        "Machine learning basics",
        created_at=datetime(2024, 3, 20, 14, 15, 0)
    )
    index.add_document(
        "Data science guide",
        created_at=datetime(2024, 6, 10, 9, 0, 0)
    )
    
    result = index.search_date_range(
        datetime(2024, 1, 1),
        datetime(2024, 3, 31),
        'created_at'
    )
    expected = BitMap([0, 1])
    assert result == expected, f"Expected {list(expected)}, got {list(result)}"


def test_search_date_range_q1_2024():
    index = InvertedIndex(
        preprocessor=TextPreprocessor(language='english'),
        use_lsm=False,
        enable_date_index=True
    )
    
    index.add_document("January doc", created_at=datetime(2024, 1, 15))
    index.add_document("February doc", created_at=datetime(2024, 2, 20))
    index.add_document("March doc", created_at=datetime(2024, 3, 25))
    index.add_document("April doc", created_at=datetime(2024, 4, 10))
    
    result = index.search_date_range(
        datetime(2024, 1, 1),
        datetime(2024, 3, 31, 23, 59, 59),
        'created_at'
    )
    expected = BitMap([0, 1, 2])
    assert result == expected, f"Expected {list(expected)}, got {list(result)}"


def test_inverted_index_with_start_end_dates():
    index = InvertedIndex(
        preprocessor=TextPreprocessor(language='english'),
        use_lsm=False,
        enable_date_index=True
    )
    
    index.add_document(
        "Project Alpha",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 6, 30)
    )
    index.add_document(
        "Project Beta",
        start_date=datetime(2024, 3, 1),
        end_date=None
    )
    index.add_document(
        "Project Gamma",
        start_date=datetime(2024, 7, 1),
        end_date=datetime(2024, 12, 31)
    )
    
    assert index.next_doc_id == 3


def test_search_valid_in_range():
    index = InvertedIndex(
        preprocessor=TextPreprocessor(language='english'),
        use_lsm=False,
        enable_date_index=True
    )
    
    index.add_document(
        "Project 1",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 6, 30)
    )
    
    index.add_document(
        "Project 2",
        start_date=datetime(2024, 3, 1),
        end_date=None
    )
    
    index.add_document(
        "Project 3",
        start_date=datetime(2024, 7, 1),
        end_date=datetime(2024, 12, 31)
    )
    
    result = index.search_valid_in_range(
        datetime(2024, 5, 1),
        datetime(2024, 5, 31)
    )
    expected = BitMap([0, 1])
    assert result == expected, f"Expected {list(expected)}, got {list(result)}"
    
    result = index.search_valid_in_range(
        datetime(2024, 8, 1),
        datetime(2024, 8, 31)
    )
    expected = BitMap([1, 2])
    assert result == expected, f"Expected {list(expected)}, got {list(result)}"


def test_search_appeared_in_range():
    index = InvertedIndex(
        preprocessor=TextPreprocessor(language='english'),
        use_lsm=False,
        enable_date_index=True
    )
    
    index.add_document(
        "Project 1",
        start_date=datetime(2024, 1, 15),
        end_date=datetime(2024, 6, 30)
    )
    index.add_document(
        "Project 2",
        start_date=datetime(2024, 3, 20),
        end_date=None
    )
    index.add_document(
        "Project 3",
        start_date=datetime(2024, 7, 10),
        end_date=datetime(2024, 12, 31)
    )
    
    result = index.search_appeared_in_range(
        datetime(2024, 1, 1),
        datetime(2024, 3, 31)
    )
    expected = BitMap([0, 1])
    assert result == expected, f"Expected {list(expected)}, got {list(result)}"
    
    result = index.search_appeared_in_range(
        datetime(2024, 7, 1),
        datetime(2024, 12, 31)
    )
    expected = BitMap([2])
    assert result == expected, f"Expected {list(expected)}, got {list(result)}"

def test_combined_text_and_date_search():
    index = InvertedIndex(
        preprocessor=TextPreprocessor(language='english'),
        use_lsm=False,
        enable_date_index=True
    )
    
    index.add_document(
        "Python programming tutorial",
        created_at=datetime(2024, 1, 15)
    )
    index.add_document(
        "Python machine learning",
        created_at=datetime(2024, 6, 20)
    )
    index.add_document(
        "Java programming guide",
        created_at=datetime(2024, 2, 10)
    )
    
    python_docs = index.search("python")
    assert python_docs == BitMap([0, 1]), f"Expected [0, 1], got {list(python_docs)}"
    
    q1_docs = index.search_date_range(
        datetime(2024, 1, 1),
        datetime(2024, 3, 31),
        'created_at'
    )
    assert q1_docs == BitMap([0, 2]), f"Expected [0, 2], got {list(q1_docs)}"
    
    result = python_docs & q1_docs
    expected = BitMap([0])
    assert result == expected, f"Expected {list(expected)}, got {list(result)}"


def test_boolean_with_dates_simple():
    index = InvertedIndex(
        preprocessor=TextPreprocessor(language='english'),
        use_lsm=False,
        enable_date_index=True
    )
    
    index.add_document(
        "Python tutorial",
        created_at=datetime(2024, 1, 15)
    )
    index.add_document(
        "Python guide",
        created_at=datetime(2024, 6, 20)
    )
    index.add_document(
        "Java tutorial",
        created_at=datetime(2024, 2, 10)
    )
    
    result = index.search_boolean_with_dates(
        "python AND date_range(created_at, 2024-01-01, 2024-03-31)"
    )
    expected = BitMap([0])
    assert result == expected, f"Expected {list(expected)}, got {list(result)}"


def test_boolean_with_valid_in():
    index = InvertedIndex(
        preprocessor=TextPreprocessor(language='english'),
        use_lsm=False,
        enable_date_index=True
    )
    
    index.add_document(
        "Active project",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 6, 30)
    )
    index.add_document(
        "Ongoing project",
        start_date=datetime(2024, 3, 1),
        end_date=None
    )
    index.add_document(
        "Future project",
        start_date=datetime(2024, 7, 1),
        end_date=datetime(2024, 12, 31)
    )
    
    result = index.search_boolean_with_dates(
        "project AND valid_in(2024-05-01, 2024-05-31)"
    )
    expected = BitMap([0, 1])
    assert result == expected, f"Expected {list(expected)}, got {list(result)}"


def test_boolean_with_appeared_in():
    index = InvertedIndex(
        preprocessor=TextPreprocessor(language='english'),
        use_lsm=False,
        enable_date_index=True
    )
    
    index.add_document(
        "Q1 project",
        start_date=datetime(2024, 1, 15),
        end_date=datetime(2024, 6, 30)
    )
    index.add_document(
        "Q2 project",
        start_date=datetime(2024, 4, 10),
        end_date=None
    )
    index.add_document(
        "Q3 project",
        start_date=datetime(2024, 7, 5),
        end_date=datetime(2024, 12, 31)
    )
    
    result = index.search_boolean_with_dates(
        "project AND appeared_in(2024-01-01, 2024-06-30)"
    )
    expected = BitMap([0, 1])
    assert result == expected, f"Expected {list(expected)}, got {list(result)}"


def main():
    try:
        cleanup_test_data()
        
        print("="*50)
        print("Running Date Search Tests")
        print("="*50)
        
        test_bit_sliced_add_and_equals()
        test_bit_sliced_greater_equal()
        test_bit_sliced_less_equal()
        test_bit_sliced_range()
        
        test_date_index_add_and_search()
        test_date_index_null_handling()
        
        test_inverted_index_with_created_at()
        test_search_date_range_q1_2024()
        
        test_inverted_index_with_start_end_dates()
        test_search_valid_in_range()
        test_search_appeared_in_range()
        
        test_combined_text_and_date_search()
        test_boolean_with_dates_simple()
        test_boolean_with_valid_in()
        test_boolean_with_appeared_in()
        
        print("\n" + "="*50)
        print("ALL DATE SEARCH TESTS PASSED!")
        print("="*50)
        
    except AssertionError as e:
        print(f"\n TEST FAILED: {e}")
        raise
    
    finally:
        cleanup_test_data()


if __name__ == "__main__":
    main()