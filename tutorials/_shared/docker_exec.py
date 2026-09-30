"""Generic Docker-based grading primitives shared by the swe-smith, swe-gym, and multi-swe-rl
tutorials — the raw docker-run / sentinel-parsing mechanics all three need, written once instead
of copy-pasted three times.

Enforces two hard-won correctness rules — see each tutorial's own TUTORIAL.md for the real
incidents that taught them:

1. A container's own exit code is NEVER trusted as the pass/fail signal. The script run inside the
   container must echo an explicit sentinel line as its last action; anything else (dead daemon,
   missing image, a script that crashed before reporting) is an environment/harness problem, not a
   verdict on the candidate solution.
2. The grading script is streamed over stdin, never passed as a `bash -c "<script>"` argv string —
   Linux caps a single argv element at 131,072 bytes (`MAX_ARG_STRLEN`), and a large patch easily
   exceeds that.
"""
from __future__ import annotations

import base64
import shlex
import subprocess
from dataclasses import dataclass

Outcome = str  # "pass" | "fail" | "error_harness" | "error_timeout"

_TAG_TO_OUTCOME = {"PASS": "pass", "FAIL": "fail", "HARNESS": "error_harness"}

_DAEMON_MARKERS = (
    "cannot connect to the docker daemon",
    "is the docker daemon running",
    "failed to connect",
    "permission denied while trying to connect",
)
_IMAGE_MARKERS = ("manifest unknown", "pull access denied", "unable to find image", "no such image")
_OUTPUT_TAIL_CHARS = 2000


@dataclass(frozen=True)
class GradeResult:
    outcome: Outcome
    detail: str = ""


def sentinel(nonce: str) -> str:
    return f"TUTORIAL_GRADE_{nonce}:"


def report_cmd(nonce: str, tag: str, detail: str = "") -> str:
    """Shell snippet a grading script runs as its LAST action: echo the sentinel line, then exit 0
    regardless of `tag` — the sentinel line itself carries the real verdict, so the container's own
    exit code is irrelevant once one has been emitted."""
    single_line_detail = " ".join(detail.split())  # collapse embedded newlines to one line
    line = sentinel(nonce) + tag + (f":{single_line_detail}" if single_line_detail else "")
    return f"echo {shlex.quote(line)}; exit 0"


def write_file_cmd(text: str, dest: str) -> str:
    """Base64 transport for writing arbitrary text (a patch, a test file) into the container —
    avoids depending on stdin (already used for the script itself) or shell-quoting content whose
    shape we don't control (a real patch can contain any byte)."""
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return f"echo {encoded} | base64 -d > {shlex.quote(dest)}"


def classify(nonce: str, returncode: int, stdout: str, stderr: str) -> GradeResult:
    """Pure function, no subprocess calls — scans combined stdout+stderr for the LAST matching
    sentinel line (not the first: a candidate's own printed output could otherwise forge an earlier
    one), and only falls back to inspecting the raw output for daemon/image trouble when no
    sentinel was ever emitted at all."""
    prefix = sentinel(nonce)
    combined = stdout + "\n" + stderr
    matched: str | None = None
    for line in combined.splitlines():
        line = line.strip()
        if line.startswith(prefix):
            matched = line[len(prefix):]

    if matched is not None:
        tag, _, detail = matched.partition(":")
        if tag in _TAG_TO_OUTCOME:
            return GradeResult(outcome=_TAG_TO_OUTCOME[tag], detail=detail)
        return GradeResult(outcome="error_harness", detail=f"malformed sentinel tag: {matched!r}")

    tail = combined.strip()[-_OUTPUT_TAIL_CHARS:]
    lowered = tail.lower()
    if any(marker in lowered for marker in _DAEMON_MARKERS):
        return GradeResult(outcome="error_harness", detail=f"docker daemon unreachable: {tail}")
    if any(marker in lowered for marker in _IMAGE_MARKERS):
        return GradeResult(outcome="error_harness", detail=f"image unavailable: {tail}")
    return GradeResult(outcome="error_harness", detail=f"no sentinel found (exit {returncode}): {tail}")


def run(
    image: str, script: str, nonce: str, timeout_seconds: int, volumes: dict[str, str] | None = None,
) -> GradeResult:
    """Runs `script` inside `image` over stdin and classifies the result via its sentinel line.
    `volumes` (host path -> container path) is optional, e.g. for mounting a build cache."""
    volume_args = [
        arg for host_path, container_path in (volumes or {}).items()
        for arg in ("-v", f"{host_path}:{container_path}")
    ]
    container_name = f"tutorial-grade-{nonce}"
    try:
        proc = subprocess.run(
            ["docker", "run", "--rm", "-i", "--name", container_name, *volume_args, image,
             "bash", "-c", "cat > /tmp/grade.sh && exec bash /tmp/grade.sh"],
            input=script,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        # Killing the `docker run` CLIENT process does NOT stop the container — the daemon owns its
        # lifecycle independently. Without an explicit kill, a timed-out container keeps running.
        kill_container(container_name)
        return GradeResult(outcome="error_timeout", detail=f"exceeded {timeout_seconds}s")
    except FileNotFoundError:
        return GradeResult(outcome="error_harness", detail="docker binary not found on PATH")
    except OSError as e:
        return GradeResult(outcome="error_harness", detail=f"OSError launching docker: {e}")

    return classify(nonce, proc.returncode, proc.stdout, proc.stderr)


def kill_container(name: str) -> None:
    """Best-effort `docker kill` — never raises, since the caller is already on its way to
    returning an error_timeout and a failed kill must not turn that into a crash."""
    try:
        subprocess.run(["docker", "kill", name], capture_output=True, text=True, timeout=60, check=False)
    except Exception:
        pass
