"""Data ingestion exceptions."""


class DataIngestionError(Exception):
    """Base exception for data ingestion failures."""


class DataValidationError(DataIngestionError):
    """Raised when source data cannot satisfy the expected schema."""
