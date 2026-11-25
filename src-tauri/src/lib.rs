use axum::{
    extract::State as AxumState,
    routing::{get, post},
    Json, Router,
};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    fs,
    io::{BufRead, BufReader, Write},
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    time::Duration,
};
use tauri::{AppHandle, Emitter, Manager};
use tokio::sync::mpsc;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message as WsMessage;
use tower_http::cors::{Any, CorsLayer};

const AI_WORKER_PORT: u16 = 38479;
const BRIDGE_PORT: u16 = 38478;
const PROJECT_ROOT_FALLBACK: &str = "/Users/tuanda/Projects/VinaNote-Offline";

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct NoteEntry {
    pub id: String,
    pub speaker: String,
    pub source: String,
    pub text: String,
    pub translation: Option<String>,
    pub time_label: String,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct SavedNote {
    pub id: String,
    pub title: String,
    pub kind: String, // "meeting" | "video" | "browser"
    pub source_url: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub summary: String,
    pub items: Vec<NoteEntry>,
}

struct LiveCaptureHandle {
    audiocap_stdin: Arc<Mutex<Option<std::process::ChildStdin>>>,
    audiocap_child: Arc<Mutex<Option<Child>>>,
    ws_tx: mpsc::UnboundedSender<String>,
}

pub struct AppState {
    ai_worker: Mutex<Option<Child>>,
    live_capture: Mutex<Option<LiveCaptureHandle>>,
    app_handle: AppHandle,
}

fn resolve_project_root() -> PathBuf {
    if let Ok(env_root) = std::env::var("VINANOTE_ROOT") {
        let p = PathBuf::from(env_root);
        if p.exists() {
            return p;
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        if cwd.join("ai-worker").exists() {
            return cwd;
        }
        if let Some(parent) = cwd.parent() {
            if parent.join("ai-worker").exists() {
                return parent.to_path_buf();
            }
        }
    }
    PathBuf::from(PROJECT_ROOT_FALLBACK)
}

fn resolve_audiocap_binary(app: &AppHandle) -> PathBuf {
    if let Ok(res_dir) = app.path().resource_dir() {
        let bundled = res_dir.join("resources").join("vinanote-audiocap");
        if bundled.exists() {
            return bundled;
        }
        let direct = res_dir.join("vinanote-audiocap");
        if direct.exists() {
            return direct;
        }
    }
    resolve_project_root()
        .join("src-tauri")
        .join("resources")
        .join("vinanote-audiocap")
}

fn notes_dir(app: &AppHandle) -> PathBuf {
    let base = app
        .path()
        .app_data_dir()
        .unwrap_or_else(|_| PathBuf::from("/Users/tuanda/Library/Application Support/local.vinanote.offline"));
    let dir = base.join("notes");
    let _ = fs::create_dir_all(&dir);
    dir
}

async fn is_worker_healthy() -> bool {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(800))
        .build();
    if let Ok(c) = client {
        if let Ok(resp) = c
            .get(format!("http://127.0.0.1:{}/health", AI_WORKER_PORT))
            .send()
            .await
        {
            return resp.status().is_success();
        }
    }
    false
}

fn spawn_ai_worker_process() -> Result<Child, String> {
    let root = resolve_project_root();
    let py = root.join("ai-worker").join(".venv").join("bin").join("python");
    let script = root.join("ai-worker").join("server.py");
    if !py.exists() || !script.exists() {
        return Err(format!(
            "Không tìm thấy môi trường AI Worker tại: {} / {}",
            py.display(),
            script.display()
        ));
    }
    Command::new(py)
        .arg(script)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(AI_WORKER_PORT.to_string())
        .current_dir(root.join("ai-worker"))
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("Không khởi chạy được AI Worker: {}", e))
}

#[tauri::command]
async fn ensure_ai_worker(state: tauri::State<'_, Arc<AppState>>) -> Result<Value, String> {
    if !is_worker_healthy().await {
        {
            let mut guard = state.ai_worker.lock().unwrap();
            if guard.is_none() {
                let child = spawn_ai_worker_process()?;
                *guard = Some(child);
            }
        }
        for _ in 0..35 {
            tokio::time::sleep(Duration::from_millis(300)).await;
            if is_worker_healthy().await {
                break;
            }
        }
    }
    let resp = reqwest::get(format!("http://127.0.0.1:{}/models/status", AI_WORKER_PORT))
        .await
        .map_err(|e| format!("AI Worker chưa phản hồi: {}", e))?;
    resp.json::<Value>()
        .await
        .map_err(|e| format!("Lỗi đọc trạng thái AI Worker: {}", e))
}

#[tauri::command]
fn check_audio_permissions(app: AppHandle) -> Result<Value, String> {
    let bin = resolve_audiocap_binary(&app);
    if !bin.exists() {
        return Err(format!("Chưa có bộ thu âm tại {}", bin.display()));
    }
    let out = Command::new(bin)
        .arg("--check")
        .output()
        .map_err(|e| format!("Lỗi kiểm tra quyền thu âm: {}", e))?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    for line in stdout.lines() {
        if let Ok(v) = serde_json::from_str::<Value>(line) {
            return Ok(v);
        }
    }
    Ok(json!({"type": "permissions", "microphone": "unknown", "system_audio": "unknown"}))
}

#[derive(Deserialize)]
struct StartLiveArgs {
    system_audio: bool,
    microphone: bool,
    language: String,
    style: String,
    translate: bool,
    speak_vi: bool,
    tts_voice: String,
}

#[tauri::command]
async fn start_live_capture(
    args: StartLiveArgs,
    state: tauri::State<'_, Arc<AppState>>,
) -> Result<(), String> {
    // Stop any previous capture session cleanly first
    stop_live_capture_inner(&state);

    // Connect to AI Worker WebSocket
    let ws_url = format!("ws://127.0.0.1:{}/ws/live", AI_WORKER_PORT);
    let (ws_stream, _) = connect_async(&ws_url)
        .await
        .map_err(|e| format!("Không kết nối được WebSocket AI Engine: {}", e))?;
    let (mut ws_write, mut ws_read) = ws_stream.split();

    let (ws_tx, mut ws_rx) = mpsc::unbounded_channel::<String>();

    // Send initial config to AI Worker
    let cfg_msg = json!({
        "type": "config",
        "language": args.language,
        "style": args.style,
        "translate": args.translate,
        "speak_vi": args.speak_vi,
        "tts_voice": args.tts_voice,
    })
    .to_string();
    let _ = ws_tx.send(cfg_msg);

    // Forward outgoing WS messages
    tokio::spawn(async move {
        while let Some(msg) = ws_rx.recv().await {
            if ws_write.send(WsMessage::Text(msg)).await.is_err() {
                break;
            }
        }
    });

    // Forward incoming AI Worker events (transcript, translation, tts_audio, error) to React UI
    let app_for_ws = state.app_handle.clone();
    tokio::spawn(async move {
        while let Some(Ok(msg)) = ws_read.next().await {
            if let WsMessage::Text(txt) = msg {
                if let Ok(val) = serde_json::from_str::<Value>(&txt) {
                    let _ = app_for_ws.emit("live-ai-event", val);
                }
            }
        }
    });

    // Spawn native Swift ScreenCaptureKit + Microphone helper
    let bin = resolve_audiocap_binary(&state.app_handle);
    if !bin.exists() {
        return Err(format!("Không tìm thấy binary thu âm: {}", bin.display()));
    }

    let mut cmd = Command::new(bin);
    cmd.arg("--exclude-pid")
        .arg(std::process::id().to_string());
    if args.system_audio {
        cmd.arg("--system");
    }
    if args.microphone {
        cmd.arg("--mic");
    }
    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("Không chạy được bộ thu âm macOS: {}", e))?;

    let child_stdin = child.stdin.take();
    let child_stdout = child
        .stdout
        .take()
        .ok_or_else(|| "Không mở được luồng stdout thu âm".to_string())?;

    let ws_tx_for_audio = ws_tx.clone();
    let app_for_audio = state.app_handle.clone();

    std::thread::spawn(move || {
        let reader = BufReader::new(child_stdout);
        for line in reader.lines().map_while(Result::ok) {
            if let Ok(v) = serde_json::from_str::<Value>(&line) {
                match v.get("type").and_then(|t| t.as_str()) {
                    Some("audio") => {
                        let src = v.get("source").and_then(|s| s.as_str()).unwrap_or("system");
                        let rms = v.get("rms").and_then(|r| r.as_f64()).unwrap_or(0.0);
                        let _ = app_for_audio.emit(
                            "audio-level",
                            json!({ "source": src, "rms": rms }),
                        );
                        let _ = ws_tx_for_audio.send(line);
                    }
                    Some("error") | Some("status") | Some("ready") => {
                        let _ = app_for_audio.emit("capture-status", v);
                    }
                    _ => {}
                }
            }
        }
    });

    let handle = LiveCaptureHandle {
        audiocap_stdin: Arc::new(Mutex::new(child_stdin)),
        audiocap_child: Arc::new(Mutex::new(Some(child))),
        ws_tx,
    };
    *state.live_capture.lock().unwrap() = Some(handle);
    Ok(())
}

fn stop_live_capture_inner(state: &Arc<AppState>) {
    let taken = state.live_capture.lock().unwrap().take();
    if let Some(h) = taken {
        let _ = h.ws_tx.send(json!({ "type": "flush" }).to_string());
        if let Some(mut stdin) = h.audiocap_stdin.lock().unwrap().take() {
            let _ = writeln!(stdin, "{{\"cmd\":\"stop\"}}");
            let _ = stdin.flush();
        }
        if let Some(mut child) = h.audiocap_child.lock().unwrap().take() {
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(350));
                let _ = child.kill();
                let _ = child.wait();
            });
        }
    }
}

#[tauri::command]
fn stop_live_capture(state: tauri::State<'_, Arc<AppState>>) -> Result<(), String> {
    stop_live_capture_inner(&state);
    Ok(())
}

#[tauri::command]
fn update_live_config(
    config: Value,
    state: tauri::State<'_, Arc<AppState>>,
) -> Result<(), String> {
    if let Some(h) = state.live_capture.lock().unwrap().as_ref() {
        if let Some(cmd_str) = config.get("audiocap_cmd").and_then(|c| c.as_str()) {
            if let Some(stdin) = h.audiocap_stdin.lock().unwrap().as_mut() {
                let _ = writeln!(stdin, "{{\"cmd\":\"{}\"}}", cmd_str);
                let _ = stdin.flush();
            }
        } else {
            let mut msg = config;
            msg["type"] = json!("config");
            let _ = h.ws_tx.send(msg.to_string());
        }
    }
    Ok(())
}

#[tauri::command]
fn list_saved_notes(app: AppHandle) -> Result<Vec<SavedNote>, String> {
    let dir = notes_dir(&app);
    let mut notes = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().and_then(|e| e.to_str()) == Some("json") {
                if let Ok(raw) = fs::read_to_string(&p) {
                    if let Ok(note) = serde_json::from_str::<SavedNote>(&raw) {
                        notes.push(note);
                    }
                }
            }
        }
    }
    notes.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    Ok(notes)
}

#[tauri::command]
fn save_note_disk(note: SavedNote, app: AppHandle) -> Result<SavedNote, String> {
    let dir = notes_dir(&app);
    let file_path = dir.join(format!("{}.json", note.id));
    let serialized = serde_json::to_string_pretty(&note).map_err(|e| e.to_string())?;
    fs::write(file_path, serialized).map_err(|e| format!("Không lưu được ghi chú: {}", e))?;
    let _ = app.emit("notes-updated", &note.id);
    Ok(note)
}

#[tauri::command]
fn delete_note_disk(id: String, app: AppHandle) -> Result<(), String> {
    let file_path = notes_dir(&app).join(format!("{}.json", id));
    if file_path.exists() {
        fs::remove_file(file_path).map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[tauri::command]
fn get_extension_path() -> String {
    resolve_project_root()
        .join("browser-extension")
        .to_string_lossy()
        .to_string()
}

// ---------------- Local Browser Extension Bridge (127.0.0.1:38478) ----------------

#[derive(Deserialize)]
struct SaveBrowserNotePayload {
    title: String,
    source_url: Option<String>,
    items: Vec<NoteEntry>,
}

async fn start_browser_bridge_server(state: Arc<AppState>) {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let router = Router::new()
        .route(
            "/v1/status",
            get(|AxumState(_st): AxumState<Arc<AppState>>| async move {
                let worker_ok = is_worker_healthy().await;
                Json(json!({
                    "ok": true,
                    "app": "VinaNote Offline",
                    "version": "1.0.0",
                    "offline": true,
                    "ai_worker_port": AI_WORKER_PORT,
                    "ai_worker_ready": worker_ok,
                }))
            }),
        )
        .route(
            "/v1/save-note",
            post(
                |AxumState(st): AxumState<Arc<AppState>>,
                 Json(payload): Json<SaveBrowserNotePayload>| async move {
                    let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
                    let note = SavedNote {
                        id: uuid::Uuid::new_v4().to_string(),
                        title: if payload.title.trim().is_empty() {
                            "Ghi chú Video Trình duyệt".to_string()
                        } else {
                            payload.title
                        },
                        kind: "browser".to_string(),
                        source_url: payload.source_url,
                        created_at: now.clone(),
                        updated_at: now,
                        summary: String::new(),
                        items: payload.items,
                    };
                    let dir = notes_dir(&st.app_handle);
                    let path = dir.join(format!("{}.json", note.id));
                    if let Ok(raw) = serde_json::to_string_pretty(&note) {
                        let _ = fs::write(path, raw);
                    }
                    let _ = st.app_handle.emit("browser-note-saved", &note);
                    if let Some(win) = st.app_handle.get_webview_window("main") {
                        let _ = win.show();
                        let _ = win.set_focus();
                    }
                    Json(json!({ "ok": true, "note_id": note.id }))
                },
            ),
        )
        .layer(cors)
        .with_state(state);

    if let Ok(listener) = tokio::net::TcpListener::bind(format!("127.0.0.1:{}", BRIDGE_PORT)).await {
        let _ = axum::serve(listener, router).await;
    }
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            let shared = Arc::new(AppState {
                ai_worker: Mutex::new(None),
                live_capture: Mutex::new(None),
                app_handle: app.handle().clone(),
            });
            app.manage(shared.clone());

            // Auto-start Python Offline AI Engine on boot
            let boot_state = shared.clone();
            tauri::async_runtime::spawn(async move {
                if !is_worker_healthy().await {
                    if let Ok(child) = spawn_ai_worker_process() {
                        *boot_state.ai_worker.lock().unwrap() = Some(child);
                    }
                }
                start_browser_bridge_server(boot_state).await;
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            ensure_ai_worker,
            check_audio_permissions,
            start_live_capture,
            stop_live_capture,
            update_live_config,
            list_saved_notes,
            save_note_disk,
            delete_note_disk,
            get_extension_path,
        ])
        .build(tauri::generate_context!())
        .expect("Lỗi khởi tạo ứng dụng VinaNote Offline")
        .run(|app_handle, event| {
            if let tauri::RunEvent::Exit = event {
                if let Some(st) = app_handle.try_state::<Arc<AppState>>() {
                    stop_live_capture_inner(&st);
                    if let Some(mut worker) = st.ai_worker.lock().unwrap().take() {
                        let _ = worker.kill();
                    }
                }
            }
        });
}
