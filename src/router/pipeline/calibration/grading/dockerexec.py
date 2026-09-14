"""Shared Docker-based grading infrastructure — one `docker run`, one shell script piped over
stdin, one sentinel line the script echoes as its last action before exiting 0. `fail` is reachable
only via an in-container `FAIL` sentinel; anything else (a dead daemon, a missing image, a crashed
script) is OUR environment's fault and classifies as `error_harness`, never `fail`. The script is
streamed over stdin rather than passed as a `bash -c` argv element, avoiding Linux's ~128KB argv
size cap. The **nonce** (caller-supplied, e.g. `uuid4().hex`) exists because the candidate's own
code is arbitrary — a fixed sentinel string would be forgeable by a patch that simply prints it.
"""
from __future__ import annotations

import base64
import logging
import re
import shlex
import subprocess
from collections import OrderedDict

from .base import GradeResult

logger = logging.getLogger(__name__)

DOCKER_TIMEOUT_SECONDS = 300  # image pull (if not cached) + container run

# Bounds how many distinct pulled images stay resident at once (see touch_image).
_MAX_CACHED_IMAGES = 15
_recently_used_images: OrderedDict[str, None] = OrderedDict()

_TAG_TO_OUTCOME: dict[str, str] = {"PASS": "pass", "FAIL": "fail", "HARNESS": "error_harness"}

# Substrings observed in real `docker run` stderr/stdout when the daemon or the image, not the
# in-container script, is the thing that failed — checked only when no sentinel was ever emitted.
_DAEMON_MARKERS = (
    "cannot connect to the docker daemon",
    "is the docker daemon running",
    "failed to connect",
    "permission denied while trying to connect",
)
_IMAGE_MARKERS = ("manifest unknown", "pull access denied", "unable to find image", "no such image")

_OUTPUT_TAIL_CHARS = 2000  # enough to see the shape of a real error; short enough not to flood logs


def sentinel(nonce: str) -> str:
    return f"ROUTER_GRADE_{nonce}:"


def report_cmd(nonce: str, tag: str, detail: str = "") -> str:
    """The shell snippet an in-container script uses as its last action: echo the sentinel line,
    then exit 0 regardless of `tag` — the sentinel line itself carries the real outcome, so the
    container's own exit code is irrelevant once one has been emitted. `detail` is collapsed to a
    single line (embedded newlines would otherwise look like additional, unrelated output lines to
    `classify`'s line-by-line scan)."""
    single_line_detail = " ".join(detail.split())
    line = sentinel(nonce) + tag + (f":{single_line_detail}" if single_line_detail else "")
    return f"echo {shlex.quote(line)}; exit 0"


def write_file_cmd(text: str, dest: str) -> str:
    """Base64 transport for writing arbitrary text (a patch, a test file) into the container
    without depending on stdin (already used for the script itself) or shell-quoting content we
    don't control."""
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return f"echo {encoded} | base64 -d > {shlex.quote(dest)}"


def pytest_collect_then_run(
    node_ids: list[str],
    nonce: str,
    repo_dir: str,
    conda_activate: str,
    fail_detail: str = "pytest reported failures",
) -> str:
    """Shared by swegym.py and swesmith.py's `_pytest_script` — both pass every declared
    FAIL_TO_PASS/PASS_TO_PASS node id to pytest in a single batch, and pytest fails the WHOLE
    invocation if even one id can't be collected, losing all signal over one bad id. Runs a cheap
    `--collect-only` pass first to exclude uncollectable ids, then the real pytest pass exactly
    once against only the survivors, folding the excluded-id count into the reported detail."""
    ids_file, collect_log, valid_file = "/tmp/node_ids.txt", "/tmp/collect.log", "/tmp/valid_ids.txt"
    prefix = sentinel(nonce)
    harness_uncollectable = report_cmd(nonce, "HARNESS", "pytest could not collect the specified test ids")
    ids_text = "\n".join(node_ids) + "\n"
    return (
        f"{write_file_cmd(ids_text, ids_file)}\n"
        f"readarray -t IDS < {ids_file}\n"
        # Collection must run from the SAME cwd (repo_dir) and env (conda_activate) as the real
        # execution pass below — pytest resolves relative node-id args against getcwd() into the
        # absolute form its own "ERROR: not found: <path>" line reports, so a mismatched cwd here
        # would make every id in step 2 below fail to match, silently excluding everything.
        f'{conda_activate} && cd {repo_dir} && python -m pytest --collect-only -q "${{IDS[@]}}" > {collect_log} 2>&1\n'
        f"> {valid_file}\n"
        f'for id in "${{IDS[@]}}"; do\n'
        # pytest reports an unresolvable id one of two ways: "not found: <repo_dir>/<id>"
        # (absolute path — the FILE exists, the specific test/class within it doesn't) or "file or
        # directory not found: <id>" (relative, id exactly as given — the file itself doesn't
        # exist). Both must be checked, or a genuinely uncollectable id can be silently kept as
        # "valid".
        f'  if grep -qxF "ERROR: not found: {repo_dir}/$id" {collect_log} || '
        f'grep -qxF "ERROR: file or directory not found: $id" {collect_log}; then :; '
        f'else echo "$id" >> {valid_file}; fi\n'
        "done\n"
        f"EXCLUDED=$(( $(wc -l < {ids_file}) - $(wc -l < {valid_file}) ))\n"
        f"if [ ! -s {valid_file} ]; then\n"
        f"  {harness_uncollectable}\n"
        "else\n"
        f"  readarray -t VALID < {valid_file}\n"
        f'  {conda_activate} && cd {repo_dir} && python -m pytest -q "${{VALID[@]}}"\n'
        "  PYTEST_EXIT=$?\n"
        '  if [ "$EXCLUDED" -gt 0 ]; then EXCL_NOTE="$EXCLUDED id(s) excluded as uncollectable"; else EXCL_NOTE=""; fi\n'
        "  if [ $PYTEST_EXIT -eq 0 ]; then\n"
        f'    if [ -n "$EXCL_NOTE" ]; then echo "{prefix}PASS:$EXCL_NOTE"; else echo "{prefix}PASS"; fi\n'
        "  elif [ $PYTEST_EXIT -eq 4 ] || [ $PYTEST_EXIT -eq 5 ]; then\n"
        f'    echo "{prefix}HARNESS:pytest could not collect the specified test ids"\n'
        "  else\n"
        f'    if [ -n "$EXCL_NOTE" ]; then echo "{prefix}FAIL:{fail_detail} ($EXCL_NOTE)"; else echo "{prefix}FAIL:{fail_detail}"; fi\n'
        "  fi\n"
        "  exit 0\n"
        "fi\n"
    )


_DIFF_GIT_HEADER_RE = re.compile(r"^diff --git a/(\S+) b/\S+", re.MULTILINE)


def diff_touched_paths(diff_text: str) -> list[str]:
    """Every file path a unified diff touches, parsed from its `diff --git a/<path> b/<path>`
    headers. Used to build `apply_patch_or_fail_cmd`'s `exclude_paths`."""
    return sorted(set(_DIFF_GIT_HEADER_RE.findall(diff_text)))


def apply_patch_or_fail_cmd(
    nonce: str, patch_path: str, fail_detail: str = "candidate patch failed to apply",
    exclude_paths: list[str] | None = None,
) -> str:
    """`git apply <patch_path> || FAIL`, folding git's own real error message into the detail
    instead of discarding it for a static string. Used for the CANDIDATE's own patch only —
    test_patch/bug_patch/gold-patch applies stay on their existing static HARNESS messages.

    `exclude_paths` (swegym.py/multiswerl.py only, via `--exclude=<path>`) skips any hunk targeting
    a file the task's own `test_patch` already touches — an agent's own worktree never has
    `test_patch` applied, so a candidate's diff is captured against a file state that no longer
    matches once grading applies `test_patch` first; a correct fix never needs to touch a
    `test_patch` file, so excluding those paths can never drop anything the fix actually required."""
    err_file = "/tmp/apply_err.txt"
    prefix = sentinel(nonce)
    exclude_flags = "".join(f" --exclude={shlex.quote(path)}" for path in (exclude_paths or []))
    return (
        f"git apply{exclude_flags} {patch_path} 2>{err_file} || {{\n"
        f"  ERR=$(tr '\\n' ' ' < {err_file} | cut -c1-500)\n"
        f'  echo "{prefix}FAIL:{fail_detail}: $ERR"\n'
        "  exit 0\n"
        "}\n"
    )


def classify(nonce: str, returncode: int, stdout: str, stderr: str) -> GradeResult:
    """PURE — no subprocess calls, so this is the part regression tests exercise directly."""
    prefix = sentinel(nonce)
    combined = stdout + "\n" + stderr
    matched: str | None = None
    for line in combined.splitlines():
        line = line.strip()
        if line.startswith(prefix):
            matched = line[len(prefix):]  # last matching line wins — no `break`

    if matched is not None:
        tag, _, detail = matched.partition(":")
        if tag in _TAG_TO_OUTCOME:
            return GradeResult(outcome=_TAG_TO_OUTCOME[tag], detail=detail)
        # A sentinel prefix with an unrecognized tag is still "our script tried to report and
        # produced something malformed" — a harness bug, not silently falling through to `fail`.
        return GradeResult(outcome="error_harness", detail=f"malformed sentinel tag: {matched!r}")

    # No sentinel anywhere ⇒ our script never got to report, so classify the *docker* failure
    # instead of defaulting to `fail` (the bug this module exists to fix).
    tail = combined.strip()[-_OUTPUT_TAIL_CHARS:]
    lowered = tail.lower()
    if any(marker in lowered for marker in _DAEMON_MARKERS):
        return GradeResult(outcome="error_harness", detail=f"docker daemon unreachable: {tail}")
    if any(marker in lowered for marker in _IMAGE_MARKERS):
        return GradeResult(outcome="error_harness", detail=f"image unavailable: {tail}")
    return GradeResult(outcome="error_harness", detail=f"no sentinel found (exit {returncode}): {tail}")


def run(
    image: str,
    script: str,
    nonce: str,
    timeout_seconds: int = DOCKER_TIMEOUT_SECONDS,
    volumes: dict[str, str] | None = None,
) -> GradeResult:
    """Runs `script` inside `image` over stdin (constant-size argv — fixes the E2BIG risk) and
    classifies the result. Writing the script to a file before executing it (rather than piping
    straight into `bash -s`) also stops any in-container command from accidentally consuming
    script bytes off stdin. `volumes` (host path -> container path) is optional and additive —
    multiswerl.py uses it to mount a persistent Go build cache."""
    volume_args = [arg for host_path, container_path in (volumes or {}).items() for arg in ("-v", f"{host_path}:{container_path}")]
    # Named so a timed-out container can actually be found and killed (see below). The nonce is
    # already unique per call and hex-only, so it's a valid container name with no collision risk.
    container_name = f"router-grade-{nonce}"
    logger.info(f"docker run starting: image={image} container={container_name} timeout={timeout_seconds}s")
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
        # Killing the `docker run` CLIENT does not stop the container — the daemon owns its
        # lifecycle, so without this explicit kill a timed-out container keeps running in the
        # background.
        kill_container(container_name)
        return GradeResult(outcome="error_timeout", detail=f"exceeded {timeout_seconds}s")
    except FileNotFoundError:
        return GradeResult(outcome="error_harness", detail="docker binary not found on PATH")
    except OSError as e:
        # Catches E2BIG and any other launch-time OS failure instead of crashing the whole
        # calibration run over a single oversized task.
        return GradeResult(outcome="error_harness", detail=f"OSError launching docker: {e}")

    return classify(nonce, proc.returncode, proc.stdout, proc.stderr)


def kill_container(name: str) -> None:
    """Best-effort `docker kill`. Never raises — the caller is already on its way to returning a
    result (an `error_timeout`), and a failed kill must not turn that into a crash. `--rm` then
    reaps the stopped container on its own."""
    try:
        subprocess.run(["docker", "kill", name], capture_output=True, text=True, timeout=60, check=False)
    except Exception:
        logger.debug(f"failed to kill container {name}", exc_info=True)


def cleanup_image(image: str) -> None:
    """Best-effort `docker rmi`. Never raises — a cleanup failure (image still referenced, already
    removed, daemon hiccup) must not turn a completed grading result into a crash; it just means
    marginally more disk pressure for that run."""
    try:
        subprocess.run(["docker", "rmi", "-f", image], capture_output=True, text=True, timeout=60, check=False)
    except Exception:
        logger.debug(f"failed to clean up image {image}", exc_info=True)


def touch_image(image: str) -> None:
    """Call after a grading run against `image` finishes. Marks it as most-recently-used and, only
    once more than `_MAX_CACHED_IMAGES` distinct images are resident, evicts the least-recently-used
    one — never `image` itself, since it's the most likely to be needed again soon. Bounds resident
    disk to a fixed ceiling while keeping recently-used images warm, instead of re-pulling a
    multi-GB image every call or letting the cache grow unbounded."""
    if image in _recently_used_images:
        _recently_used_images.move_to_end(image)
    else:
        _recently_used_images[image] = None

    while len(_recently_used_images) > _MAX_CACHED_IMAGES:
        oldest, _ = _recently_used_images.popitem(last=False)
        cleanup_image(oldest)
