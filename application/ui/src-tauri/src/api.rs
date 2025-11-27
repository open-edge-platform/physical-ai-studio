use reqwest::Client;
use serde_json::Value;

#[tauri::command]
pub async fn py_api(
    method: String,
    endpoint: String,
    payload: Option<Value>,
) -> Result<Value, String> {
    let client = Client::new();
    let url = format!("http://127.0.0.1:9100/{}", endpoint);

    let request = match method.as_str() {
        "GET" => client.get(&url),
        "POST" => {
            let req = client.post(&url);
            if let Some(data) = &payload {
                req.json(data)
            } else {
                req
            }
        }
        "PUT" => client.put(&url),
        "DELETE" => client.delete(&url),
        _ => return Err(format!("Unsupported HTTP method: {}", method)),
    };

    let request = if let Some(data) = payload {
        request.json(&data)
    } else {
        request
    };

    let response = request
        .send()
        .await
        .map_err(|e| e.to_string())?
        .json::<Value>()
        .await
        .map_err(|e| e.to_string())?;

    Ok(response)
}