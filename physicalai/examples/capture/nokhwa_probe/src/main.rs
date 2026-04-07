use std::env;
use std::error::Error;
use std::sync::mpsc;
use std::time::Duration;

use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{RequestedFormat, RequestedFormatType};
use nokhwa::{native_api_backend, query, Camera};

#[cfg(target_os = "macos")]
fn ensure_macos_initialized() -> Result<(), Box<dyn Error>> {
    let (tx, rx) = mpsc::channel();
    nokhwa::nokhwa_initialize(move |ok| {
        let _ = tx.send(ok);
    });

    let init_ok = rx
        .recv_timeout(Duration::from_secs(20))
        .map_err(|e| format!("Timed out waiting for macOS camera permission callback: {e}"))?;

    if !init_ok || !nokhwa::nokhwa_check() {
        return Err("Nokhwa initialization failed on macOS. Check camera permission in System Settings.".into());
    }

    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn ensure_macos_initialized() -> Result<(), Box<dyn Error>> {
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    ensure_macos_initialized()?;

    let selected_index = env::args().nth(1).and_then(|v| v.parse::<usize>().ok());

    let backend = native_api_backend().ok_or("No native Nokhwa backend available on this platform")?;
    println!("native backend: {:?}", backend);

    let infos = query(backend)?;
    if infos.is_empty() {
        println!("No cameras discovered by Nokhwa.");
        return Ok(());
    }

    println!("\nDiscovered cameras ({}):", infos.len());
    for (i, info) in infos.iter().enumerate() {
        println!("- camera[{}]", i);
        println!("  index: {:?}", info.index());
        println!("  human_name: {}", info.human_name());
        println!("  description: {}", info.description());
        println!("  misc: {}", info.misc());
    }

    let probe_idx = selected_index.unwrap_or(0);
    let selected = infos
        .get(probe_idx)
        .ok_or_else(|| format!("Invalid camera index {} (found {})", probe_idx, infos.len()))?;

    println!("\nInspecting camera[{}]: {}", probe_idx, selected.human_name());

    let requested = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
    let mut camera = Camera::new(selected.index().clone(), requested)?;

    println!("camera.info(): {:?}", camera.info());
    println!("camera.camera_format(): {:?}", camera.camera_format());

    match camera.compatible_camera_formats() {
        Ok(formats) => {
            println!("\ncompatible_camera_formats ({}):", formats.len());
            for fmt in formats {
                println!("  {:?}", fmt);
            }
        }
        Err(e) => println!("\ncompatible_camera_formats: error: {e}"),
    }

    match camera.compatible_fourcc() {
        Ok(fourccs) => {
            println!("\ncompatible_fourcc ({}):", fourccs.len());
            for fourcc in fourccs {
                println!("  fourcc: {:?}", fourcc);
                match camera.compatible_list_by_resolution(fourcc) {
                    Ok(map) => {
                        for (res, fps) in map {
                            println!("    {:?} -> {:?}", res, fps);
                        }
                    }
                    Err(e) => println!("    compatible_list_by_resolution error: {e}"),
                }
            }
        }
        Err(e) => println!("\ncompatible_fourcc: error: {e}"),
    }

    match camera.camera_controls() {
        Ok(controls) => {
            println!("\ncamera_controls ({}):", controls.len());
            for control in controls {
                println!("  {:?}", control);
            }
        }
        Err(e) => println!("\ncamera_controls: error: {e}"),
    }

    Ok(())
}
