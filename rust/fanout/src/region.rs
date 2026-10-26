//! Refusing to fan out over a bucket in another region than the workers:
//! reading across regions is billed per GB, about $0.02, more than the query
//! itself.

use std::collections::HashMap;

use aws_config::BehaviorVersion;
use datafusion::error::{DataFusionError, Result};
use object_store::aws::resolve_bucket_region;
use object_store::ClientOptions;
use tokio::sync::{Mutex, OnceCell};
use url::Url;

/// Why reading `bucket` from workers in `workers` costs extra, if it does.
pub fn problem(bucket: &str, region: &str, workers: &str) -> Option<String> {
    (region != workers).then(|| {
        format!(
            "s3://{bucket} is in {region} but the workers are in {workers}, and reading \
             across regions costs about $0.02 a GB: run the workers in {region}, or allow \
             cross-region reads"
        )
    })
}

/// Fails if any of the S3 `locations` is in another region than the workers.
pub(crate) async fn check(locations: &[String]) -> Result<()> {
    let Some(workers) = workers_region().await else {
        return Ok(());
    };
    for location in locations {
        let Some(bucket) = s3_bucket(location) else {
            continue;
        };
        if let Some(problem) = problem(&bucket, &bucket_region(&bucket).await?, workers) {
            return Err(DataFusionError::Plan(problem));
        }
    }
    Ok(())
}

fn s3_bucket(location: &str) -> Option<String> {
    let url = Url::parse(location).ok()?;
    (url.scheme() == "s3").then(|| url.host_str().map(str::to_string))?
}

/// Where the Lambda client invokes, from the same configuration it loads.
async fn workers_region() -> Option<&'static str> {
    static REGION: OnceCell<Option<String>> = OnceCell::const_new();
    REGION
        .get_or_init(|| async {
            let config = aws_config::defaults(BehaviorVersion::latest()).load().await;
            config.region().map(|r| r.to_string())
        })
        .await
        .as_deref()
}

async fn bucket_region(bucket: &str) -> Result<String> {
    static REGIONS: Mutex<Option<HashMap<String, String>>> = Mutex::const_new(None);
    let mut regions = REGIONS.lock().await;
    let regions = regions.get_or_insert_with(HashMap::new);
    if let Some(region) = regions.get(bucket) {
        return Ok(region.clone());
    }
    let region = resolve_bucket_region(bucket, &ClientOptions::default()).await?;
    regions.insert(bucket.to_string(), region.clone());
    Ok(region)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bucket_in_another_region_is_a_problem() {
        assert!(problem("b", "us-east-1", "eu-west-1")
            .unwrap()
            .contains("us-east-1"));
        assert_eq!(problem("b", "eu-west-1", "eu-west-1"), None);
    }

    #[test]
    fn only_s3_locations_have_buckets() {
        assert_eq!(s3_bucket("s3://era5/group").as_deref(), Some("era5"));
        assert_eq!(s3_bucket("gs://era5/group"), None);
        assert_eq!(s3_bucket("/data/era5"), None);
    }
}
