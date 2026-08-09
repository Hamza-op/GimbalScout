//! Durable same-directory file replacement.
//!
//! Writers first create and fsync a uniquely named sibling, then replace the
//! destination in one filesystem operation. Keeping this logic in one place
//! prevents settings, cache, and XML writers from briefly deleting the last
//! known-good file before the replacement is ready.

use std::ffi::OsString;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::{AppError, AppResult};

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub fn write_bytes(path: &Path, bytes: &[u8]) -> AppResult<()> {
    let (tmp_path, mut file) = create_temp_file(path)?;
    let result = (|| {
        file.write_all(bytes).map_err(|source| AppError::Io {
            path: tmp_path.clone(),
            source,
        })?;
        file.sync_all().map_err(|source| AppError::Io {
            path: tmp_path.clone(),
            source,
        })?;
        drop(file);
        replace_file(&tmp_path, path).map_err(|source| AppError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        sync_parent(path)?;
        Ok(())
    })();

    if result.is_err() {
        let _ = fs::remove_file(&tmp_path);
    }
    result
}

fn create_temp_file(path: &Path) -> AppResult<(PathBuf, fs::File)> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let file_name = path
        .file_name()
        .map(|name| name.to_os_string())
        .unwrap_or_else(|| OsString::from("output"));

    for _ in 0..32 {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let mut tmp_name = OsString::from(".");
        tmp_name.push(&file_name);
        tmp_name.push(format!(".{}.{}.tmp", std::process::id(), sequence));
        let tmp_path = parent.join(tmp_name);
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp_path)
        {
            Ok(file) => return Ok((tmp_path, file)),
            Err(source) if source.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(source) => {
                return Err(AppError::Io {
                    path: tmp_path,
                    source,
                });
            }
        }
    }

    Err(AppError::Message(format!(
        "could not allocate a temporary file beside {}",
        path.display()
    )))
}

#[cfg(windows)]
fn replace_file(from: &Path, to: &Path) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH, MoveFileExW,
    };

    let from = from
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let to = to
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    // SAFETY: both buffers are owned, NUL-terminated UTF-16 strings and stay
    // alive for the duration of the Win32 call. The flags require no extra
    // pointers or callback state.
    let ok = unsafe {
        MoveFileExW(
            from.as_ptr(),
            to.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if ok == 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(not(windows))]
fn replace_file(from: &Path, to: &Path) -> std::io::Result<()> {
    fs::rename(from, to)
}

#[cfg(unix)]
fn sync_parent(path: &Path) -> AppResult<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::File::open(parent)
        .and_then(|dir| dir.sync_all())
        .map_err(|source| AppError::Io {
            path: parent.to_path_buf(),
            source,
        })
}

#[cfg(not(unix))]
fn sync_parent(_path: &Path) -> AppResult<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replaces_existing_file_without_leaving_temp_files() {
        let root = std::env::temp_dir()
            .join("video-tool-atomic-write-test")
            .join(std::process::id().to_string());
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let target = root.join("state.json");
        fs::write(&target, b"old").unwrap();

        write_bytes(&target, b"new").unwrap();

        assert_eq!(fs::read(&target).unwrap(), b"new");
        assert_eq!(fs::read_dir(&root).unwrap().count(), 1);
    }
}
