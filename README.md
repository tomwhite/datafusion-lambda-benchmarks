# datafusion-lambda-benchmarks

How fast can we query ERA5 Zarr data using SQL and the [map-lambda](https://github.com/tomwhite/map-lambda) framework?

**The monthly area-weighted global mean temperature over 85 years of hourly
ERA5 (745,128 chunks, 3.1 TB uncompressed) takes ~88s and costs ~$2.17**

**The hourly climatology of every grid cell over the same ERA5 record, written to
Zarr on S3, takes ~136s and costs ~$3.07**

Both answers match xarray for a sample, and the global means match those
Copernicus publishes from ERA5 to within 0.001°C for all but one of the ~1000
months.

## How it works

A query is planned by [DataFusion](https://datafusion.apache.org/), over Zarr
arrays flattened into tables by [zarr-datafusion](https://github.com/stratoscale-io/zarr-datafusion).

A new physical operator `MapLambdaZarrExec` is used to run partial
aggregates on Lambda workers. For the global means, the partial aggregate results
for each worker are streamed back to the client which completes the rest of the plan.

For the climatology, each worker processes all of a group's rows, so it runs
the final aggregate, and writes the output itself.

This project is set up for running benchmarks, it is not a general purpose SQL engine.

## Dependencies

Both map-lambda and zarr-datafusion are built from pinned versions on GitHub.

zarr-datafusion comes from the `lambda-zarr-write` branch of a fork,
<https://github.com/tomwhite/zarr-datafusion>, pinned to a commit, until its
Zarr writes are merged upstream.

## Running locally

First start `cargo lambda watch` from the `rust/` directory, building first,
because `watch` compiles a function on its first request, which would outlast
the dispatcher's invoke timeout:

```shell
cd rust && cargo build && cargo lambda watch
```

Then, in another terminal, run the tests:

```shell
uv run pytest
cd rust && cargo test
```

## Running on AWS

The data is in us-east-1, so the workers must be there too (a query refuses
to run otherwise). `OUTPUT_BUCKET` is only needed for the climatology, which
the workers write to:

```shell
OUTPUT_BUCKET=<bucket> ./aws/setup-roles.sh   # once - wait a few seconds for IAM to apply
AWS_REGION=us-east-1 ./aws/deploy.sh
./aws/teardown.sh      # when you're finished
```

The client plans against S3 with credentials from `AWS_*` variables,
so export as follows:

```shell
eval "$(aws configure export-credentials --format env)"; export AWS_REGION=us-east-1
```

Each benchmark below saves its run to `results/`, with a timeline of its
invocations.

## Global mean temperature

`benchmarks/demo.py` computes the monthly area-weighted global mean of `t2m`
over [Earthmover's ERA5](https://registry.opendata.aws/earthmover-era5/),
dataset in Icechunk format. The full dataset goes back to 1940, but the
script accepts a time range to allow you to query part of it.

Start with a few months to check everything is working properly:

```shell
uv run python benchmarks/demo.py --start 2020-01-01 --end 2020-04-01
```

It should complete in a few seconds.

Then run the whole dataset, which costs about $2. The concurrency leaves room
under the account's default limit of 1,000:

```shell
uv run python benchmarks/demo.py --start 1940-01-01 --end 2025-01-01 --concurrency 990
```

In one run, the 1,020 months took 88.2s and cost approximately $2.17:
31,047 invocations of 24 hourly chunks.

<img src="./era5-1940-2024-c990-timeline.svg">

### Checking the results

```shell
# one month against xarray, reading about 3 GB from us-east-1
uv run benchmarks/demo_check.py $(ls results/demo-*.json | tail -1) --month 2020-02
# every month against Copernicus's published daily means, charted beside the run
uv run python benchmarks/copernicus_check.py $(ls results/demo-*.json | tail -1)
```

The month matches xarray to 9 decimal places, with identical extremes. Every
month from 1940 to 2022 agrees with Copernicus within about 0.0006°C, which
rounds each day to 0.001°C. Earthmover's copy of 2023 and 2024 differs a
little more, up to 0.0013°C in July 2024. This discrepancy was not explored further.

<img src="./era5-1940-2024-copernicus.svg">

### The plan

`--explain` prints the query's physical plan, then the fragment that each
fan-out sends to the workers. It doesn't run the query.

```shell
uv run python benchmarks/demo.py --start 1940-01-01 --end 2025-01-01 --concurrency 990 --explain
```

The client runs everything down to the `MapLambdaZarrExec`, which stands in
for the partial aggregate and the scan below it: its 745,128 chunks, those of
the window that pruning kept, go to 31,047 invocations of 24.

```
SortPreservingMergeExec: [month@0 ASC NULLS LAST]
  SortExec: expr=[month@0 ASC NULLS LAST], preserve_partitioning=[true]
    ProjectionExec: expr=[date_trunc(Utf8("month"),era5.valid_time)@0 as month, sum(era5.t2m * cos(radians(era5.latitude)))@1 / sum(cos(radians(era5.latitude)))@2 as mean_t2m, min(era5.t2m)@3 as min_t2m, max(era5.t2m)@4 as max_t2m, count(Int64(1))@5 as n]
      AggregateExec: mode=FinalPartitioned, gby=[date_trunc(Utf8("month"),era5.valid_time)@0 as date_trunc(Utf8("month"),era5.valid_time)], aggr=[sum(era5.t2m * cos(radians(era5.latitude))), sum(cos(radians(era5.latitude))), min(era5.t2m), max(era5.t2m), count(Int64(1))]
        RepartitionExec: partitioning=Hash([date_trunc(Utf8("month"),era5.valid_time)@0], 8), input_partitions=1
          MapLambdaZarrExec: function=fragment, chunks=745128, invocations=31047

fragment:
AggregateExec: mode=Partial, gby=[date_trunc(month, CAST(valid_time@1 AS Timestamp(ns))) as date_trunc(Utf8("month"),era5.valid_time)], aggr=[sum(era5.t2m * cos(radians(era5.latitude))), sum(cos(radians(era5.latitude))), min(era5.t2m), max(era5.t2m), count(Int64(1))]
  ProjectionExec: expr=[cos(radians(CAST(latitude@1 AS Float64))) as __common_expr_1, valid_time@0 as valid_time, t2m@2 as t2m]
    FilterExec: valid_time@0 >= -946771200000000 AND valid_time@0 < 1735689600000000
      ZarrExec: path=s3://earthmover-icechunk-era5/icechunkV2, filters=[valid_time BETWEEN -946771200000000 AND 1735689600000000]
```

Each worker binds the fragment's `ZarrExec` to its own chunks, and returns
one partial row per month it touches.

## Hourly climatology

`benchmarks/climatology.py` computes the mean `t2m` of every grid cell for
each hour of the day, over the same ERA5 in its time-series layout
(`single/temporal`), where a chunk is a "pencil": a year of hours (8,736) over
a 12 x 12 tile. The output is 24 x 721 x 1440 cells, written by the workers to
a Zarr store with a chunk a tile:

```sql
COPY (
  SELECT latitude, longitude,
         date_part('hour', CAST(valid_time AS TIMESTAMP)) AS hour,
         avg(t2m) AS clim, count(*) AS n
  FROM era5
  WHERE valid_time >= TIMESTAMP '{start}' AND valid_time < TIMESTAMP '{end}'
  GROUP BY 1, 2, 3
) TO 's3://<bucket>/climatology.zarr' STORED AS ZARR OPTIONS ('chunks' '24,12,12')
```

Each of the 7,320 invocations takes one tile's whole stack of pencils, so its
rows are final: it finishes the aggregate, writes its tile's chunk, and
returns only a count.

By default it runs over a week, which is useful for checking it works first
(costs about $0.12).

```shell
uv run python benchmarks/climatology.py s3://<bucket>/climatology-week.zarr
```

Then the whole record:

```shell
uv run python benchmarks/climatology.py s3://<bucket>/climatology.zarr --start 1940-01-01 --end 2026-01-01
```

In one run, the whole record took 135.8s and cost approximately $3.07: 7,320
invocations of 87 pencils, in about 7.4 waves at a concurrency of 990. The
cost includes about $0.07 for the workers' ~14,640 writes to the output, which
the run's report leaves out.

<img src="./era5-climatology-c990-timeline.svg">

### Checking the results

```shell
# one tile, 57.25 to 60°N and 0 to 2.75°E, against xarray reading the same pencils
uv run --with xarray --with "numcodecs[pcodec]" --with obstore python benchmarks/climatology_check.py \
  --output s3://<bucket>/climatology.zarr --start 1940-01-01 --end 2026-01-01
```

The tile matches xarray exactly. The check reads the tile's chunk alone, in
about 2 seconds.

## AI usage

Code and tests were written with AI assistance from Claude.
