"""Apache TsFile format support"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import set_module

from pandas.core.api import DataFrame

from pandas.io.common import stringify_path

if TYPE_CHECKING:
    from pandas._typing import (
        FilePath,
    )


# Mapping from pandas/numpy dtypes to TsFile data types
def _get_tsfile_dtype(dtype):
    """
    Map pandas dtype to TsFile TSDataType.

    Parameters
    ----------
    dtype : numpy dtype or pandas dtype
        The dtype to convert.

    Returns
    -------
    TSDataType
        The corresponding TsFile data type.
    """
    tsfile = import_optional_dependency("tsfile")
    TSDataType = tsfile.TSDataType

    dtype_str = str(dtype)

    if dtype_str in ("int8", "int16", "int32", "Int8", "Int16", "Int32"):
        return TSDataType.INT32
    elif dtype_str in ("int64", "Int64"):
        return TSDataType.INT64
    elif dtype_str in ("float32", "Float32"):
        return TSDataType.FLOAT
    elif dtype_str in ("float64", "Float64"):
        return TSDataType.DOUBLE
    elif dtype_str in ("bool", "boolean"):
        return TSDataType.BOOLEAN
    elif dtype_str.startswith("datetime64"):
        return TSDataType.TIMESTAMP
    elif dtype_str in ("object", "string", "str"):
        return TSDataType.STRING
    elif dtype_str.startswith("bytes"):
        return TSDataType.BLOB
    else:
        # Default to STRING for unsupported types
        return TSDataType.STRING


def _convert_value(value, tsfile_dtype):
    """
    Convert a value to the appropriate type for TsFile.

    Parameters
    ----------
    value : Any
        The value to convert.
    tsfile_dtype : TSDataType
        The target TsFile data type.

    Returns
    -------
    Any
        The converted value.
    """
    tsfile = import_optional_dependency("tsfile")
    TSDataType = tsfile.TSDataType

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    if tsfile_dtype == TSDataType.INT32:
        return int(value)
    elif tsfile_dtype == TSDataType.INT64:
        return int(value)
    elif tsfile_dtype == TSDataType.FLOAT:
        return float(value)
    elif tsfile_dtype == TSDataType.DOUBLE:
        return float(value)
    elif tsfile_dtype == TSDataType.BOOLEAN:
        return bool(value)
    elif tsfile_dtype == TSDataType.TIMESTAMP:
        # Convert datetime to timestamp in nanoseconds
        if hasattr(value, "value"):
            return int(value.value)
        return int(value)
    elif tsfile_dtype == TSDataType.STRING:
        return str(value)
    elif tsfile_dtype == TSDataType.BLOB:
        if isinstance(value, bytes):
            return value
        return str(value).encode("utf-8")
    else:
        return value


def to_tsfile(
    df: DataFrame,
    path: FilePath,
    *,
    table_name: str = "default_table",
) -> None:
    """
    Write a DataFrame to the TsFile format.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to write.
    path : str or path object
        String or path object (implementing ``os.PathLike[str]``) specifying
        the file path to write the TsFile to.
    table_name : str, default "default_table"
        Name of the table in the TsFile.

    Returns
    -------
    None

    See Also
    --------
    read_tsfile : Read a TsFile into a DataFrame.
    DataFrame.to_parquet : Write a DataFrame to a parquet file.

    Notes
    -----
    * This function requires the `tsfile <https://pypi.org/project/tsfile/>`_
      library.
    * TsFile is a columnar file format designed for time-series data.
    * The TsFile format automatically adds a 'time' column using row indices
      as timestamps.

    Examples
    --------
    >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    >>> df.to_tsfile("data.tsfile")  # doctest: +SKIP
    """
    tsfile = import_optional_dependency(
        "tsfile", extra="tsfile is required for TsFile support."
    )
    from tsfile import (
        ColumnSchema,
        TableSchema,
        Tablet,
        TsFileTableWriter,
    )

    if not isinstance(df, DataFrame):
        raise ValueError("to_tsfile only supports IO with DataFrames")

    path = stringify_path(path)

    # Build column schemas
    column_names = list(df.columns)
    tsfile_dtypes = [_get_tsfile_dtype(df[col].dtype) for col in column_names]

    columns = [
        ColumnSchema(str(col), dtype) for col, dtype in zip(column_names, tsfile_dtypes)
    ]
    schema = TableSchema(table_name, columns)

    # Write to TsFile
    with TsFileTableWriter(path, schema) as writer:
        # Create tablet
        tablet = Tablet(
            [str(col) for col in column_names], tsfile_dtypes, max_row_num=len(df)
        )
        tablet.set_table_name(table_name)

        # Set timestamps using index if it's datetime, otherwise use range
        if hasattr(df.index, "asi8") and df.index.dtype.kind == "M":
            # DateTime index - use nanoseconds
            timestamps = df.index.asi8.tolist()
        else:
            # Use row indices as timestamps
            timestamps = list(range(len(df)))
        tablet.set_timestamp_list(timestamps)

        # Add values
        for row_idx in range(len(df)):
            for col_idx, col in enumerate(column_names):
                value = df.iloc[row_idx, col_idx]
                converted_value = _convert_value(value, tsfile_dtypes[col_idx])
                if converted_value is not None:
                    tablet.add_value_by_index(col_idx, row_idx, converted_value)

        writer.write_table(tablet)


@set_module("pandas")
def read_tsfile(
    path: FilePath,
    *,
    table_name: str | None = None,
    columns: list[str] | None = None,
    start_time: int | None = None,
    end_time: int | None = None,
) -> DataFrame:
    """
    Load a TsFile object from the file path, returning a DataFrame.

    Parameters
    ----------
    path : str or path object
        String or path object (implementing ``os.PathLike[str]``) specifying
        the file path to read the TsFile from.
    table_name : str, optional
        Name of the table to read from the TsFile. If None, the first table
        found in the schema will be used.
    columns : list of str, optional
        If not None, only these columns will be read from the file.
    start_time : int, optional
        Start timestamp for time range filtering.
    end_time : int, optional
        End timestamp for time range filtering.

    Returns
    -------
    DataFrame
        DataFrame based on the TsFile.

    See Also
    --------
    DataFrame.to_tsfile : Write a DataFrame to a TsFile.
    read_parquet : Read a parquet file into a DataFrame.

    Notes
    -----
    * This function requires the `tsfile <https://pypi.org/project/tsfile/>`_
      library.
    * TsFile is a columnar file format designed for time-series data.

    Examples
    --------
    >>> df = pd.read_tsfile("data.tsfile")  # doctest: +SKIP
    """
    import_optional_dependency(
        "tsfile", extra="tsfile is required for TsFile support."
    )
    from tsfile import to_dataframe

    path = stringify_path(path)

    # Build kwargs for to_dataframe
    kwargs: dict = {}
    if table_name is not None:
        kwargs["table_name"] = table_name
    if columns is not None:
        kwargs["column_names"] = columns
    if start_time is not None:
        kwargs["start_time"] = start_time
    if end_time is not None:
        kwargs["end_time"] = end_time

    result = to_dataframe(path, **kwargs)

    return result
