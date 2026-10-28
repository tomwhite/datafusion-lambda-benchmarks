use lambda_runtime::Error;

#[tokio::main]
async fn main() -> Result<(), Error> {
    map_lambda_dispatcher::run().await
}
