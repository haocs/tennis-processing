"""Storage checks for independently configured private input and shared output."""
from pathlib import Path
import os


def check_library_storage(cfg):
    root = Path(cfg['library_root']).resolve()
    private = Path(cfg.get('input_root', cfg.get('nas_root', '.'))).resolve()
    if private.is_relative_to(root) or root.is_relative_to(private):
        raise RuntimeError('Private input and published library must be separate, non-nested folders')
    mount = cfg.get('library_mount_point')
    if mount and not os.path.ismount(mount):
        raise RuntimeError('Published-library volume is not mounted')
    if not root.is_dir():
        raise RuntimeError('Published library is unavailable; initialize it with libraryctl first')
    marker = root / '.tennis-library-id'
    expected = cfg.get('library_storage_id')
    if not marker.is_file() or (expected and marker.read_text().strip() != expected):
        raise RuntimeError('Published-library storage identity is missing or does not match')
    return root


def status_destination(cfg, legacy_check):
    return check_library_storage(cfg) / 'metadata' if cfg.get('library_root') else legacy_check(cfg)


def ensure_library_storage(cfg):
    """Reconnect a configured Mac mount using saved credentials, with no password in config."""
    try:
        return check_library_storage(cfg)
    except (OSError, RuntimeError):
        mount = cfg.get('library_mount_point')
        source = cfg.get('library_mount_source')
        if mount and source and not os.path.ismount(mount):
            import subprocess
            import time
            state = Path(cfg['state_dir'])
            marker = state / 'last-library-reconnect'
            last = marker.stat().st_mtime if marker.exists() else 0
            if time.time() - last > 600:
                if not source.startswith('//') or ':' in source.split('@')[0]:
                    raise RuntimeError('Use a password-free SMB mount source')
                marker.touch()
                subprocess.run(['/sbin/mount_smbfs', '-N', source, mount], check=True, capture_output=True, timeout=45)
        return check_library_storage(cfg)
