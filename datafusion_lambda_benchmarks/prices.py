"""List prices for what a query uses: arm64 Lambda and S3 Standard requests."""

# standard US and EU regions
PRICING_AS_OF = "2026-09-08"
GB_SECOND_USD = 0.0000133334
REQUEST_USD = 0.20 / 1_000_000
S3_GET_USD = 0.0004 / 1000
# LIST is billed with PUT, COPY and POST
S3_LIST_USD = 0.005 / 1000

# Lambda bills memory in GB of 1024 MB.
MEGABYTES_PER_GIGABYTE = 1024


def gigabytes(memory_size_mb: float) -> float:
    """A function's memory size in the gigabytes Lambda bills in."""
    return memory_size_mb / MEGABYTES_PER_GIGABYTE
