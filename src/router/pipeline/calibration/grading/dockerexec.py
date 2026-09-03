"""Shared Docker-based grading infrastructure — one `docker run`, one shell script piped over
stdin, one sentinel line the script echoes as its last action before exiting 0.

Two invariants this module exists to enforce (see docs/... calibration plan for the bugs this
fixes in the code that predates it):

1. **`fail` is reachable only via an in-container `FAIL` sentinel.** Anything else — a dead
   daemon, a missing image, a crashed script that never got to report — is OUR environment's
   fault, not the candidate's, and must classify as `error_harness` (excluded from error rates),
   never `fail` (counted as a wrong answer).
2. **The script is never passed as a single argv element.** Linux caps a single argv element at
   131,072 bytes (`MAX_ARG_STRLEN`); a large patch base64'd into a `bash -c <script>` argv can
   exceed that and crash `execve` with `OSError(E2BIG)`. Streaming the script over stdin instead
   has no such limit.

The **nonce** (caller-supplied, e.g. `uuid4().hex`) exists because pytest echoes captured test
stdout and the candidate's own code is arbitrary — a fixed sentinel string would be forgeable by a
patch that simply prints it.
"""
from __future__ import annotations

import base64
import logging
import shlex
import subprocess
from collections import OrderedDict

from .base import GradeResult

logger = logging.getLogger(__name__)

DOCKER_TIMEOUT_SECONDS = 300  # image pull (if not cached) + container run

# Bounds how many distinct pulled images stay resident at once (see touch_image). See
# docs/engineering-notes.md, "Dockerexec image cache sizing".
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
    invocation (exit 4 "usage error" / 5 "no tests collected") if even one id can't be collected,
    losing all signal for the task over one bad id. See docs/engineering-notes.md, "Pytest
    collection mismatch (swegym/swesmith)".

    Splits into an explicit collect-then-execute shape instead of "run for real, retry on
    failure" so the potentially-slow execution step runs at most once, by construction — a
    `--collect-only` pass can never execute a test body, so it's cheap regardless of suite size:

    1. `--collect-only` against every declared id.
    2. Check each id individually against that pass's own `ERROR: not found: <repo_dir>/<id>`
       lines — an exact full-line match (`grep -x`), not a substring one, since a substring match
       would let one valid id be wrongly excluded just for being a literal prefix of a different,
       genuinely-bad id's line.
    3. If every id turns out uncollectable, report HARNESS immediately — no point invoking pytest
       again on an empty set.
    4. Otherwise run the REAL pytest pass exactly once, against only the survivors. Its own exit
       code still gets the same 0 / 4-or-5 / else handling as a safety net for a collection
       failure step 2 didn't happen to explain per-id (e.g. a session-wide conftest import error),
       and the excluded-id count (if any) is folded into the reported detail so it's visible
       without re-deriving it from a discarded log."""
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


def apply_patch_or_fail_cmd(nonce: str, patch_path: str, fail_detail: str = "candidate patch failed to apply") -> str:
    """`git apply <patch_path> || FAIL`, folding git's own real error message into the detail
    instead of discarding it for a static string. Used for the CANDIDATE's own patch only —
    test_patch/bug_patch/gold-patch applies stay on their existing static HARNESS messages (a
    dataset/harness problem regardless of git's specific error there). Without the real error, a
    candidate's apply failure gives no way to tell "the diff is malformed" from "the diff doesn't
    match this baseline" after the fact.

    `2>{err_file}` isolates git's stderr from the rest of the script's own stdout — `tr`+`cut`
    collapses it to one line and caps it at 500 chars, matching `report_cmd`'s own single-line
    convention (an embedded newline would otherwise look like additional, unrelated output lines
    to `classify`'s line-by-line scan)."""
    err_file = "/tmp/apply_err.txt"
    prefix = sentinel(nonce)
    return (
        f"git apply {patch_path} 2>{err_file} || {{\n"
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
    script bytes off stdin.

    `volumes` (host path -> container path) is optional and additive — omitted entirely by default,
    so existing callers (swesmith.py, swegym.py) are unaffected. multiswerl.py uses it to mount a
    persistent Go build cache across otherwise-fresh `--rm` containers (see its own module
    docstring for why repeated cold compiles of the same repo are the dominant cost there)."""
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
        # background. See docs/engineering-notes.md, "Dockerexec timeout kill".
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
    one — NOT `image` itself, which was just used and is the most likely of all of them to be
    needed again soon (e.g. `validate-graders` grading the same task's reference then null control
    in quick succession, or `calibrate` grading it once per model).

    Deliberately not "clean up right after every call": with calibration's models-outer/tasks-inner
    loop, that would re-pull the same multi-GB image once per model instead of once per run. An
    unbounded cache (never cleaning up at all) trades that for unbounded disk growth across a long
    run touching many distinct repos. This bounds resident disk to a fixed ceiling
    (`_MAX_CACHED_IMAGES` images) while keeping recently-used images warm."""
    if image in _recently_used_images:
        _recently_used_images.move_to_end(image)
    else:
        _recently_used_images[image] = None

    while len(_recently_used_images) > _MAX_CACHED_IMAGES:
        oldest, _ = _recently_used_images.popitem(last=False)
        cleanup_image(oldest)
