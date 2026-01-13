"""test tsfile compat"""

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm

from pandas.io.tsfile import (
    read_tsfile,
    to_tsfile,
)

try:
    import tsfile as tsfile_lib

    _HAVE_TSFILE = True
except ImportError:
    _HAVE_TSFILE = False


pytestmark = pytest.mark.skipif(not _HAVE_TSFILE, reason="tsfile is not installed")


@pytest.fixture
def temp_file(tmp_path):
    return tmp_path / "test.tsfile"


@pytest.fixture
def df_simple():
    return pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7.0, 8.0, 9.0]})


@pytest.fixture
def df_mixed_types():
    return pd.DataFrame(
        {
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
            "bool_col": [True, False, True],
        }
    )


def check_round_trip(df, temp_file, **kwargs):
    """Verify tsfile serializer and deserializer produce the same results."""
    df.to_tsfile(temp_file, **kwargs)
    result = read_tsfile(temp_file)

    # TsFile always adds a 'time' column
    assert "time" in result.columns

    # Compare the data columns (excluding 'time')
    result_cols = [col for col in result.columns if col != "time"]

    # Column names are lowercased by TsFile
    expected_cols = [col.lower() for col in df.columns]

    # Check all expected columns are present
    for col in expected_cols:
        assert col in result_cols, f"Column {col} not found in result"

    # Check data values match
    for orig_col, res_col in zip(df.columns, expected_cols):
        # Get the result column values
        result_values = result[res_col].values
        expected_values = df[orig_col].values

        orig_dtype = df[orig_col].dtype

        # Compare values
        if orig_dtype == "bool":
            np.testing.assert_array_equal(result_values, expected_values)
        elif orig_dtype.kind == "f":
            np.testing.assert_array_almost_equal(result_values, expected_values)
        else:
            np.testing.assert_array_equal(result_values, expected_values)


class TestTsFileBasic:
    def test_error_not_dataframe(self, temp_file):
        """Test that non-DataFrame objects raise an error."""
        for obj in [
            pd.Series([1, 2, 3]),
            1,
            "foo",
            pd.Timestamp("20130101"),
            np.array([1, 2, 3]),
        ]:
            msg = "to_tsfile only supports IO with DataFrames"
            with pytest.raises(ValueError, match=msg):
                to_tsfile(obj, temp_file)

    def test_simple_round_trip(self, df_simple, temp_file):
        """Test basic round trip with simple DataFrame."""
        check_round_trip(df_simple, temp_file)

    def test_mixed_types_round_trip(self, df_mixed_types, temp_file):
        """Test round trip with mixed data types."""
        check_round_trip(df_mixed_types, temp_file)

    def test_integer_types(self, temp_file):
        """Test various integer types."""
        df = pd.DataFrame(
            {
                "int8": np.array([1, 2, 3], dtype=np.int8),
                "int16": np.array([1, 2, 3], dtype=np.int16),
                "int32": np.array([1, 2, 3], dtype=np.int32),
                "int64": np.array([1, 2, 3], dtype=np.int64),
            }
        )
        check_round_trip(df, temp_file)

    def test_float_types(self, temp_file):
        """Test float types."""
        df = pd.DataFrame(
            {
                "float32": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "float64": np.array([1.0, 2.0, 3.0], dtype=np.float64),
            }
        )
        check_round_trip(df, temp_file)

    def test_custom_table_name(self, df_simple, temp_file):
        """Test writing with custom table name."""
        table_name = "my_custom_table"
        df_simple.to_tsfile(temp_file, table_name=table_name)
        result = read_tsfile(temp_file, table_name=table_name)

        # Should have 'time' column plus data columns
        assert "time" in result.columns
        assert len(result) == len(df_simple)

    def test_read_columns_filter(self, df_simple, temp_file):
        """Test reading with column filter."""
        df_simple.to_tsfile(temp_file)

        # TsFile lowercases column names
        result = read_tsfile(temp_file, columns=["a", "b"])

        # Should have 'time' column plus filtered columns
        assert "time" in result.columns
        assert "a" in result.columns
        assert "b" in result.columns
        # Column 'c' should not be present (wasn't requested)
        assert "c" not in result.columns


class TestTsFileDataFrame:
    def test_dataframe_to_tsfile_method(self, df_simple, temp_file):
        """Test DataFrame.to_tsfile method."""
        df_simple.to_tsfile(temp_file)

        result = pd.read_tsfile(temp_file)
        assert "time" in result.columns
        assert len(result) == len(df_simple)

    def test_pandas_read_tsfile(self, df_simple, temp_file):
        """Test pd.read_tsfile function."""
        to_tsfile(df_simple, temp_file)

        result = pd.read_tsfile(temp_file)
        assert "time" in result.columns
        assert len(result) == len(df_simple)


class TestTsFileLargeData:
    def test_large_dataframe(self, temp_file):
        """Test with larger DataFrame."""
        n_rows = 10000
        df = pd.DataFrame(
            {
                "int_col": np.random.randint(0, 100, n_rows),
                "float_col": np.random.random(n_rows),
            }
        )

        df.to_tsfile(temp_file)
        result = read_tsfile(temp_file)

        assert len(result) == n_rows
        assert "time" in result.columns

