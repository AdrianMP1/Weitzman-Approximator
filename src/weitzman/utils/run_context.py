
import atexit
import shutil
import socket
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

@dataclass
class RunContext:
    run_dir: Path
    run_id: str
    keep: bool
    _cleanup_registered: bool = False
    _success: bool = False

    def mark_success(self) -> None:
        self._success = True

    def register_cleanup(self) -> None:
        """

        """

        if self._cleanup_registered:
            return

        def _cleanup() -> None:
            # Policy: Delete only if keep=False and run succeeded
            if not self.keep and self._success:
                shutil.rmtree(self.run_dir, ignore_errors=True)

        atexit.register(_cleanup)
        self._cleanup_registered = True


def create_run_context(
    results_root: Path,
    prefix: str = "run",
    keep: bool = True,
    run_id: str | None = None
) -> RunContext:
    """
    Creates a unique run folder: results/runs/<run_id>/
    """

    results_root = results_root.resolve()
    runs_root = results_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    if run_id is None:
        host = socket.gethostname().split(".")[0]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{prefix}_{ts}_{host}"

    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    ctx = RunContext(run_dir=run_dir, run_id=run_id, keep=keep)
    ctx.register_cleanup()
    return ctx
