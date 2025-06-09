import numpy as np
import pytest
from pycissa.processing.cissa.cissa import Cissa

def test_cissa_init_use_32_bit_true_conversion():
    """
    Test 1: use_32_bit=True with convertible numerics.
    - Input x array with integers and float64.
    - Verify that all elements in x (as processed by initial_data_checks via Cissa instantiation) become np.float32.
    """
    t_test = np.array([1, 2, 3, 4])
    x_test_initial = [1, 2.0, np.int64(3), np.float64(4.0)]

    cissa_instance = Cissa(t=np.copy(t_test), x=np.copy(x_test_initial), use_32_bit=True)
    x_processed = cissa_instance.x

    assert x_processed.dtype == np.float32, "Array dtype should be float32"
    for val in x_processed:
        assert isinstance(val, np.float32), f"Value {val} is not float32"

def test_cissa_init_use_32_bit_false_no_conversion():
    """
    Test 2: use_32_bit=False (default).
    - Input x array with integers (e.g., np.int32) and np.float64.
    - Verify that the data types of elements in x remain unchanged (e.g. np.int32, np.float64).
    """
    t_test = np.array([1, 2, 3])

    # Test with integers
    x_int_initial = np.array([1, 2, 3], dtype=np.int32)
    cissa_int_instance = Cissa(t=np.copy(t_test), x=np.copy(x_int_initial), use_32_bit=False)
    x_int_processed = cissa_int_instance.x
    # Note: np.array([1,2,3], dtype=np.int32) when passed to np.array(x) in initial_data_checks
    # might become np.int64 by default if not handled carefully.
    # However, the key is it should NOT become float32.
    # The initial_data_checks function itself will try to make it a default numpy array (int64 or float64).
    # Let's check the type it becomes after initial_data_checks's np.array(x) call.
    # If initial_data_checks receives np.array([1,2,3], dtype=np.int32), it should keep it as is or convert to a default int type.
    # For this test, we expect the type to be preserved if it's already a NumPy array.
    assert x_int_processed.dtype == np.int32 or x_int_processed.dtype == np.int64, f"Integer array dtype {x_int_processed.dtype} was not expected int32 or int64"


    # Test with float64
    x_float64_initial = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    cissa_float64_instance = Cissa(t=np.copy(t_test), x=np.copy(x_float64_initial), use_32_bit=False)
    x_float64_processed = cissa_float64_instance.x
    assert x_float64_processed.dtype == np.float64, "Float64 array dtype should remain float64"

    # Test with mixed Python types (which numpy will homogenize)
    x_mixed_initial = [1, 2.0, 3] # Will become float64 array
    cissa_mixed_instance = Cissa(t=np.copy(t_test), x=np.copy(x_mixed_initial), use_32_bit=False)
    x_mixed_processed = cissa_mixed_instance.x
    assert x_mixed_processed.dtype == np.float64, "Mixed Python types should lead to float64 numpy array"


def test_cissa_init_use_32_bit_true_mixed_handling():
    """
    Test 3: use_32_bit=True with mixed types including non-numeric and overflow.
    - Input x array with float64, convertible string, non-convertible string,
      a value that overflows float32, and a standard integer.
    - Verify correct conversions and retentions.
    """
    t_test = np.array([1, 2, 3, 4, 5])
    # Value that results in np.inf when converted to float32
    overflow_val = np.finfo(np.float64).max

    x_test_initial = np.array([1.0, "2.0", "text", 300, overflow_val], dtype=object)

    cissa_instance = Cissa(t=np.copy(t_test), x=np.copy(x_test_initial), use_32_bit=True)
    x_processed = cissa_instance.x

    # Assertions
    assert x_processed.dtype == object, "Array dtype should be object due to mixed types"

    assert isinstance(x_processed[0], np.float32), "Value 1.0 should be float32"
    assert x_processed[0] == np.float32(1.0), "Value 1.0 incorrect"

    assert isinstance(x_processed[1], np.float32), "Value '2.0' should be float32"
    assert x_processed[1] == np.float32(2.0), "Value '2.0' incorrect"

    assert isinstance(x_processed[2], str), "Value 'text' should remain a string"
    assert x_processed[2] == "text", "Value 'text' incorrect"

    assert isinstance(x_processed[3], np.float32), "Value 300 should be float32"
    assert x_processed[3] == np.float32(300.0), "Value 300 incorrect"

    # np.float32(np.finfo(np.float64).max) results in np.inf (float32 type)
    assert isinstance(x_processed[4], np.float32), "Overflow value should be float32 (inf)"
    assert x_processed[4] == np.float32(np.inf), "Overflow value should be inf"

def test_cissa_init_use_32_bit_true_overflow_within_float32_max_edge_case():
    """
    Test with a large number that is just at the edge of float32's max representable value.
    And one that is np.finfo(np.float32).max.
    """
    t_test = np.array([1, 2])
    # Max finite float32 value
    float32_max = np.finfo(np.float32).max

    x_test_initial = np.array([float32_max, float32_max * 0.999], dtype=np.float64) # Start as float64

    cissa_instance = Cissa(t=np.copy(t_test), x=np.copy(x_test_initial), use_32_bit=True)
    x_processed = cissa_instance.x

    assert x_processed.dtype == np.float32, "Array dtype should be float32"

    assert isinstance(x_processed[0], np.float32), "float32_max should be float32"
    assert x_processed[0] == np.float32(float32_max), "float32_max value incorrect"

    assert isinstance(x_processed[1], np.float32), "Slightly less than float32_max should be float32"
    # Comparison for floating point numbers requires tolerance or careful handling
    # For this specific case, direct conversion should be exact if it's representable
    assert x_processed[1] == np.float32(float32_max * 0.999)

def test_cissa_init_use_32_bit_true_with_standard_list_input():
    """
    Test with a standard Python list as input for x.
    """
    t_test = np.array([1, 2, 3])
    x_list_initial = [10, 20.5, 30] # Python ints and floats

    cissa_instance = Cissa(t=np.copy(t_test), x=x_list_initial, use_32_bit=True)
    x_processed = cissa_instance.x

    assert x_processed.dtype == np.float32
    assert all(isinstance(val, np.float32) for val in x_processed)
    assert np.array_equal(x_processed, np.array([10.0, 20.5, 30.0], dtype=np.float32))

def test_cissa_init_use_32_bit_false_with_standard_list_input():
    """
    Test with a standard Python list as input for x and use_32_bit=False.
    """
    t_test = np.array([1, 2, 3])
    x_list_initial = [10, 20.5, 30] # Python ints and floats

    # Numpy will convert [10, 20.5, 30] to a float64 array by default.
    cissa_instance = Cissa(t=np.copy(t_test), x=x_list_initial, use_32_bit=False)
    x_processed = cissa_instance.x

    assert x_processed.dtype == np.float64
    assert np.array_equal(x_processed, np.array([10.0, 20.5, 30.0], dtype=np.float64))

def test_cissa_init_empty_input_array_with_use_32_bit_true():
    """
    Test with an empty input array for x and use_32_bit=True.
    """
    t_test = np.array([])
    x_empty_initial = np.array([])

    cissa_instance = Cissa(t=np.copy(t_test), x=np.copy(x_empty_initial), use_32_bit=True)
    x_processed = cissa_instance.x

    # np.array([]).astype(np.float32) results in dtype float32.
    assert x_processed.dtype == np.float32 or x_processed.dtype == np.float64
    # An empty array when use_32_bit is true, after passing through np.empty_like(x, dtype=object)
    # and then x.astype(np.float32) might end up float64 if x was initially float64 (empty_like preserves type for data region)
    # or float32. Let's be more specific: initial_data_checks gets x = np.array([]).
    # new_x = np.empty_like(x, dtype=object) -> new_x is np.array([], dtype=object)
    # loop is skipped. x = new_x.
    # x.astype(np.float32) -> np.array([], dtype=np.float32).
    assert x_processed.dtype == np.float32
    assert len(x_processed) == 0

def test_cissa_init_empty_input_array_with_use_32_bit_false():
    """
    Test with an empty input array for x and use_32_bit=False.
    """
    t_test = np.array([])
    x_empty_initial = np.array([], dtype=np.int32) # Give a specific type

    cissa_instance = Cissa(t=np.copy(t_test), x=np.copy(x_empty_initial), use_32_bit=False)
    x_processed = cissa_instance.x

    assert x_processed.dtype == np.int32
    assert len(x_processed) == 0
