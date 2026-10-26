use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use zarrs_storage::{
    byte_range::ByteRangeIterator, AsyncListableStorageTraits, AsyncMaybeBytesIterator,
    AsyncReadableListableStorage, AsyncReadableStorageTraits, MaybeBytes, StorageError, StoreKey,
    StoreKeys, StoreKeysPrefixes, StorePrefix,
};

/// Object store requests by S3's billing tiers: GET and HEAD, then LIST.
/// Process-wide, since remote stores are cached across invocations.
static GETS: AtomicU64 = AtomicU64::new(0);
static LISTS: AtomicU64 = AtomicU64::new(0);
/// Time spent in them, bodies included, summed over concurrent ones.
static NANOS: AtomicU64 = AtomicU64::new(0);

/// Requests made so far: GETs, LISTs, and the time spent in them.
pub(crate) fn made() -> (u64, u64, Duration) {
    (
        GETS.load(Ordering::Relaxed),
        LISTS.load(Ordering::Relaxed),
        Duration::from_nanos(NANOS.load(Ordering::Relaxed)),
    )
}

/// Held by tests that read the counters, which are process-wide.
#[cfg(test)]
pub(crate) static COUNTERS: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

async fn timed<T>(counter: &AtomicU64, request: impl Future<Output = T>) -> T {
    counter.fetch_add(1, Ordering::Relaxed);
    let start = Instant::now();
    let result = request.await;
    NANOS.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
    result
}

/// `inner`, counting and timing the requests made through it. A paginated
/// listing counts once, and a multi-range read once, though zarrs' object
/// store adapter also makes a HEAD for it and the store may coalesce or split
/// its ranges: so the counts are close, not exact.
pub(crate) fn counted(inner: AsyncReadableListableStorage) -> AsyncReadableListableStorage {
    std::sync::Arc::new(Counted(inner))
}

struct Counted(AsyncReadableListableStorage);

#[async_trait]
impl AsyncReadableStorageTraits for Counted {
    async fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        timed(&GETS, self.0.get(key)).await
    }

    async fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<AsyncMaybeBytesIterator<'a>, StorageError> {
        timed(&GETS, self.0.get_partial_many(key, byte_ranges)).await
    }

    async fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        timed(&GETS, self.0.size_key(key)).await
    }

    fn supports_get_partial(&self) -> bool {
        self.0.supports_get_partial()
    }
}

#[async_trait]
impl AsyncListableStorageTraits for Counted {
    async fn list(&self) -> Result<StoreKeys, StorageError> {
        timed(&LISTS, self.0.list()).await
    }

    async fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
        timed(&LISTS, self.0.list_prefix(prefix)).await
    }

    async fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
        timed(&LISTS, self.0.list_dir(prefix)).await
    }

    async fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
        timed(&LISTS, self.0.size_prefix(prefix)).await
    }

    async fn size(&self) -> Result<u64, StorageError> {
        timed(&LISTS, self.0.size()).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use object_store::{memory::InMemory, path::Path, ObjectStoreExt, PutPayload};
    use zarrs_object_store::AsyncObjectStore;
    use zarrs_storage::byte_range::ByteRange;

    use super::*;

    #[tokio::test]
    async fn counts_each_request_by_its_billing_tier() {
        let memory = InMemory::new();
        memory
            .put(&Path::from("a/c/0"), PutPayload::from_static(b"chunk"))
            .await
            .unwrap();
        let store = counted(Arc::new(AsyncObjectStore::new(memory)));
        let key = StoreKey::new("a/c/0").unwrap();

        let _counters = COUNTERS.lock().await;
        let before = made();
        store.get(&key).await.unwrap().unwrap();
        store
            .get_partial(&key, ByteRange::FromStart(1, Some(2)))
            .await
            .unwrap()
            .unwrap();
        store.size_key(&key).await.unwrap();
        store
            .list_prefix(&StorePrefix::new("a/").unwrap())
            .await
            .unwrap();
        let after = made();

        assert_eq!((after.0 - before.0, after.1 - before.1), (3, 1));
        assert!(after.2 > before.2);
    }
}
