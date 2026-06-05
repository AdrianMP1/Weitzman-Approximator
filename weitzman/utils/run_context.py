import socket

from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

@dataclass
class RunContext:
    run_dir: Path
    run_id: str
    _success: bool = field(default=False, init=False, repr=False)

    def mark_success(self) -> None:
        self._success = True


def create_run_context(
    results_root: Path | str,
    prefix: str = "run",
    run_id: str | None = None,
) -> RunContext:
    """
    Create a timestamped output directory under results_root/runs/.

    If run_id is provided it is used as-is; otherwise one is generated
    from the prefix, current timestamp, and hostname.
    """
    results_root = Path(results_root).resolve()
    runs_root = results_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    if run_id is None:
        host = socket.gethostname().split(".")[0]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{prefix}_{ts}_{host}"

    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    return RunContext(run_dir=run_dir, run_id=run_id)
