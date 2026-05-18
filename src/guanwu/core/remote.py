"""Remote GPU execution via SSH.

Provides ``RemoteExecutor`` for running Python scripts on a remote machine
(typically a GPU server) via SSH, and transferring files via SCP.
"""
from __future__ import annotations

import logging
import shlex
import subprocess
import textwrap
from pathlib import Path

from guanwu.core.config import RemoteConfig
from guanwu.core.errors import BlueBirdError

logger = logging.getLogger("guanwu")


class RemoteError(BlueBirdError):
    """Error during remote execution."""


class RemoteExecutor:
    """Execute Python scripts on a remote machine via SSH.

    The remote machine must be reachable via ``ssh <host>`` (configured
    in ``~/.ssh/config``).  Scripts run inside a conda environment.

    Args:
        config: ``RemoteConfig`` with host, conda_env, work_dir, etc.
    """

    def __init__(self, config: RemoteConfig) -> None:
        if not config.host:
            raise RemoteError("remote.host is not configured")
        self.host = config.host
        self.conda_env = config.conda_env
        self.work_dir = config.work_dir
        self.python = config.python
        self.conda_init = config.conda_init

    # ── Connection test ────────────────────────────────────────────

    def test_connection(self) -> dict:
        """Test SSH connectivity, conda env, and GPU availability."""
        result = {"host": self.host, "ok": False}
        try:
            out = self._ssh("hostname && nvidia-smi --query-gpu=name --format=csv,noheader | head -1", timeout=15)
            lines = out.strip().splitlines()
            result["hostname"] = lines[0] if lines else "unknown"
            result["gpu"] = lines[1].strip() if len(lines) > 1 else "none"
        except Exception as e:
            result["error"] = f"SSH failed: {e}"
            return result

        if self.conda_env:
            try:
                out = self._ssh_conda(f"{self.python} --version", timeout=15)
                result["python"] = out.strip()
            except Exception as e:
                result["error"] = f"conda env failed: {e}"
                return result

        result["ok"] = True
        return result

    # ── Script execution ───────────────────────────────────────────

    def run_script(self, script: str, timeout: int = 300) -> str:
        """Run a Python script on the remote machine.

        The script is passed via heredoc to ``python3`` inside the
        configured conda environment.

        Returns:
            stdout of the remote script.

        Raises:
            RemoteError: on non-zero exit or timeout.
        """
        self._ensure_work_dir()
        wrapped = f'{self.python} << \'BLUEBIRD_REMOTE_EOF\'\n{script}\nBLUEBIRD_REMOTE_EOF'
        return self._ssh_conda(wrapped, timeout=timeout)

    # ── File transfer ──────────────────────────────────────────────

    def upload(self, local_path: str | Path, remote_path: str | None = None) -> str:
        """Upload a local file to the remote work_dir (or a specific path).

        Returns the remote file path.
        """
        local_path = Path(local_path)
        if not local_path.exists():
            raise RemoteError(f"Local file not found: {local_path}")

        if remote_path is None:
            remote_path = f"{self.work_dir}/{local_path.name}"

        # Ensure remote directory exists
        remote_dir = str(Path(remote_path).parent)
        self._ssh(f"mkdir -p {remote_dir}", timeout=10)

        # Use rsync (more reliable for large files) with fallback to scp
        for cmd in [
            ["rsync", "-az", "--progress", str(local_path), f"{self.host}:{remote_path}"],
            ["scp", str(local_path), f"{self.host}:{remote_path}"],
        ]:
            logger.debug("upload: %s", " ".join(cmd))
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                if proc.returncode == 0:
                    logger.info("Uploaded %s -> %s:%s", local_path, self.host, remote_path)
                    return remote_path
            except FileNotFoundError:
                continue  # rsync not available, try scp
            except subprocess.TimeoutExpired:
                logger.warning("Upload timed out with %s", cmd[0])
                continue

        raise RemoteError(f"Upload failed for {local_path}")

    def download(self, remote_path: str, local_path: str | Path) -> Path:
        """Download a file from the remote machine.

        Returns the local path.
        """
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = ["scp", "-q", f"{self.host}:{remote_path}", str(local_path)]
        logger.debug("download: %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            raise RemoteError(f"scp download failed: {proc.stderr.strip()}")

        logger.info("Downloaded %s:%s -> %s", self.host, remote_path, local_path)
        return local_path

    # ── Package management ─────────────────────────────────────────

    def ensure_package(self, package: str, import_name: str | None = None) -> None:
        """Ensure a Python package is installed in the remote conda env."""
        module_name = import_name or package.replace("-", "_")
        try:
            self._ssh_conda(
                f"{self.python} -c \"import {module_name}\"",
                timeout=60,
            )
            logger.debug("Package %s already installed on %s", package, self.host)
        except RemoteError:
            logger.info("Installing %s on %s...", package, self.host)
            self._ssh_conda(f"pip install {package}", timeout=180)

    def path_exists(self, remote_path: str) -> bool:
        """Return True if a remote file or directory exists."""
        quoted = shlex.quote(remote_path)
        try:
            self._ssh(f"test -e {quoted}", timeout=30)
            return True
        except RemoteError:
            return False

    # ── Internal helpers ───────────────────────────────────────────

    def _ssh(self, command: str, timeout: int = 60) -> str:
        """Run a raw shell command over SSH."""
        cmd = ["ssh", self.host, command]
        logger.debug("ssh: %s %s", self.host, command[:120])
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            raise RemoteError(
                f"SSH command failed (exit {proc.returncode}): {proc.stderr.strip()[:500]}",
                details={"command": command[:200], "stderr": proc.stderr[:1000]},
            )
        return proc.stdout

    def _ssh_conda(self, command: str, timeout: int = 60) -> str:
        """Run a command inside the conda environment over SSH."""
        if self.conda_env:
            wrapped = (
                f"source {self.conda_init} && "
                f"conda activate {self.conda_env} && "
                f"{command}"
            )
        else:
            wrapped = command
        return self._ssh(wrapped, timeout=timeout)

    def _ensure_work_dir(self) -> None:
        """Create the remote work directory if it doesn't exist."""
        self._ssh(f"mkdir -p {self.work_dir}", timeout=10)


def get_remote_executor(config: RemoteConfig) -> RemoteExecutor | None:
    """Create a RemoteExecutor if remote is configured, else return None."""
    if not config.host:
        return None
    return RemoteExecutor(config)
