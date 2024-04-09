use parquet::errors::ParquetError;

pub fn into_parquet_error(
    err: impl Into<Box<dyn std::error::Error + Send + Sync>>,
) -> ParquetError {
    ParquetError::External(err.into())
}
