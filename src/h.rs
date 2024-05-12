#![deny(warnings)]
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModel, SentenceEmbeddingsModelType,
};

use once_cell::sync::Lazy;
use warp::Filter;
use std::sync::Mutex;

use serde_derive::{Deserialize, Serialize};
#[derive(Deserialize, Serialize)]
struct Employee {
    name: String,
}

static MODEL: Lazy<Mutex<SentenceEmbeddingsModel>> = Lazy::new(|| {
    println!("loading model");
    Mutex::new(SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL12V2)
        .create_model()
        .unwrap())
});

#[tokio::main]
async fn main() {
    // GET /hello/warp => 200 OK with body "Hello, warp!"
    let promote = warp::post()
        .and(warp::path("employees"))
        // Only accept bodies smaller than 16kb...
        .and(warp::body::content_length_limit(1024 * 16))
        .and(warp::body::json())
        .map(|employee: Employee| {
            let _ = MODEL.lock().unwrap().encode(&[&employee.name]).unwrap();
            println!("g");
            warp::reply::json(&employee)
        });
    
    warp::serve(promote).run(([127, 0, 0, 1], 3030)).await
}
