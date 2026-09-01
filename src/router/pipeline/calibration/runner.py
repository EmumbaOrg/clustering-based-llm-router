"""Invokes the Pi coding agent headlessly (`pi -p`) to produce a candidate solution for a task
against a configured model. This is the ONLY code path that calls a real agent.

The `reference`/`null` grader-validation controls (see config/models.yaml) do NOT go through this
module — they're synthesized directly by calibrate.py, source by source, without invoking Pi at
all (the "reference" solution is just the dataset's own gold value; "null" is an empty solution).
That keeps the controls fast, free, and independent of agent/model behavior, which is the whole
point of using them to validate a grader before trusting any real model's score.

Extracting "the solution" from an agent's free-form response is inherently fuzzy — we ask for a
single fenced code block and take the first one found, falling back to the whole response if none
is fenced. This is a known limitation of prompt-based extraction rather than a structured-output
contract with Pi; acceptable for this pass's completeness goal (see pipeline-python/README.md),
but expect it to hurt a real model's measured score independently of its actual coding ability.
For sources `repo_context.py` knows how to check out (swe-smith today), this is sidestepped
instead of worked around: the agent gets a real, isolated working tree and its own edit/write
tools, and the solution is captured via `git diff` on that tree rather than parsed from prose —
text extraction stays only as the fallback for a response with no repo context or no tool use.

Every call is paced by `RateLimiter` against the candidate's configured `rate_limit_rpm` (see
config/models.yaml). That paces *request count*, but Groq's free tier also caps *tokens per
minute* — observed in practice to bind much sooner than RPM for these prompt sizes, since a
rejection reserves the request's prompt + max_tokens against the budget up front rather than
waiting to see actual usage. Groq's own error message names an exact cooldown ("Please try again
in Ns"), so a TPM rejection is retried after that exact wait (`MAX_RATE_LIMIT_RETRIES` times, each
capped at `MAX_RATE_LIMIT_WAIT_SECONDS`) rather than immediately given up on. Only once retries are
exhausted does `RunResult.rate_limited` get set, routing calibrate.py to exclude the task
(`error_harness`) instead of counting a quota rejection as a wrong answer.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import re
import subprocess
import time
from pathlib import Path

from ...common.config import ModelConfig
from . import repo_context
from .grading.base import Task

logger = logging.getLogger(__name__)

_CODE_BLOCK_RE = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)

# Substrings looked for (case-insensitively) in a failed pi call's stderr to tell "the provider
# rate-limited us" apart from any other failure. Deliberately loose — providers don't share a
# wire format for this, and a false positive here only costs one `error_harness` exclusion, while
# a false negative would let a rate-limit response silently inflate a model's measured error rate.
_RATE_LIMIT_MARKERS = ("429", "rate limit", "rate_limit", "too many requests")

# Groq's rate-limit error names the exact cooldown, e.g. "Please try again in 20.19s" or
# "in 780ms" — parsed so a rejection can be retried after precisely that long instead of guessing.
_RETRY_AFTER_RE = re.compile(r"try again in\s+([\d.]+)\s*(ms|s)\b", re.IGNORECASE)

MAX_RATE_LIMIT_RETRIES = 2
# Never wait longer than this for one retry even if the server asks for more — a free-tier window
# occasionally asks for 30s+; better to exclude the task than block a whole calibration run on it.
MAX_RATE_LIMIT_WAIT_SECONDS = 45.0


def _parse_retry_after_seconds(stderr: str) -> float | None:
    match = _RETRY_AFTER_RE.search(stderr)
    if not match:
        return None
    value = float(match.group(1))
    return value / 1000.0 if match.group(2).lower() == "ms" else value


class RateLimiter:
    """Paces successive calls per model_id to at most `rpm` per minute — the free-tier limits
    documented per-model on the provider's own rate-limit page (see config/models.yaml). A
    SAFETY_MARGIN is applied on top of the documented limit since our own clock and the
    provider's rate-limit window aren't perfectly aligned; better to run slightly under the
    documented cap than to 429 right at the boundary.

    `clock`/`sleep` are injectable so tests can exercise the pacing logic without a real sleep.
    """

    SAFETY_MARGIN = 1.15

    def __init__(self, clock=time.monotonic, sleep=time.sleep):
        self._clock = clock
        self._sleep = sleep
        self._last_call_at: dict[str, float] = {}

    def wait(self, key: str, rpm: int | None) -> None:
        if not rpm:
            return
        min_interval = (60.0 / rpm) * self.SAFETY_MARGIN
        last = self._last_call_at.get(key)
        now = self._clock()
        if last is not None:
            remaining = min_interval - (now - last)
            if remaining > 0:
                self._sleep(remaining)
                now = self._clock()
        self._last_call_at[key] = now


_rate_limiter = RateLimiter()


def _looks_rate_limited(stderr: str) -> bool:
    lowered = stderr.lower()
    return any(marker in lowered for marker in _RATE_LIMIT_MARKERS)

# Per-source instructions appended to the task's own natural-language prompt, telling the agent
# what SHAPE of answer each grader expects (see the grading/*.py module docstrings for why each
# shape is what it is — e.g. bigcodebench wants a body fragment because code_prompt already ends
# mid-signature).
_INSTRUCTIONS = {
    "bigcodebench": (
        "Respond with ONLY the function body, indented, continuing directly from this signature "
        "(no import statements, no signature line, no explanation):\n\n{code_prompt}\n\n"
        "Wrap your answer in a single ```python code block."
    ),
    "ds1000": (
        "Respond with ONLY the code snippet that solves this — no explanation. "
        "Wrap your answer in a single ```python code block."
    ),
    "swe-smith": (
        "Respond with a unified diff (git apply-compatible) that fixes this, and nothing else. "
        "Wrap it in a single ```diff code block."
    ),
    "swe-gym": (
        "Respond with a unified diff (git apply-compatible) that fixes this, and nothing else. "
        "Wrap it in a single ```diff code block."
    ),
    "multi-swe-rl": (
        "Respond with a unified diff (git apply-compatible) that fixes this, and nothing else. "
        "Wrap it in a single ```diff code block."
    ),
}

# Used instead of _INSTRUCTIONS' entry when repo_context has checked out a real working tree for
# this task (see build_prompt) — the agent has read/bash/edit/write tools pointed at it, so it's
# told to use them directly rather than hand-write a diff from memory. Only sources with a
# _REPO_SOURCES entry (repo_context.py) ever reach this path.
#
# No "summarize the change or include a diff" escape hatch — confirmed empirically this session
# why that phrasing was a real bug, not just imprecise wording. There is only ONE path that ever
# reaches the grading Docker container: repo_context.extract_diff() runs `git diff` on this local
# worktree AFTER the call finishes, and THAT diff — never the agent's prose or a code block in its
# text response — is what gets shipped and `git apply`'d in a container that shares nothing else
# with this worktree. A model that took the "summarize instead" option produced plain code with no
# diff structure (confirmed on real claude-haiku-4-5 calibration rows, e.g. arrow-py/sqlfluff:
# bare function bodies, not `diff --git` output) — extract_diff() found no real edits to report,
# fell back to parsing that text, and `git apply` had nothing valid to work with. It could never
# have worked regardless of how the model formatted its answer. The fix is to remove the option
# entirely: the only real deliverable is actually editing the files.
_CONTEXT_INSTRUCTIONS = {
    "swe-smith": (
        "The repository is checked out in your current working directory, at the state before "
        "this issue was fixed. Use your available tools to explore the codebase, understand the "
        "issue, and make the necessary changes directly using your edit/write tools — actually "
        "modify the files; do not just describe or show the fix in your response. Your final "
        "message does not need to include any code or diff — the changes you make to the files "
        "are the submission. Keep your change as small and targeted as possible — modify only "
        "what's strictly necessary to fix the described issue; do not update changelogs, CI "
        "configuration, documentation, or unrelated code as part of this fix."
    ),
    "swe-gym": (
        "The repository is checked out in your current working directory, at the state before "
        "this issue was fixed. Use your available tools to explore the codebase, understand the "
        "issue, and make the necessary changes directly using your edit/write tools — actually "
        "modify the files; do not just describe or show the fix in your response. Your final "
        "message does not need to include any code or diff — the changes you make to the files "
        "are the submission. Keep your change as small and targeted as possible — modify only "
        "what's strictly necessary to fix the described issue; do not update changelogs, CI "
        "configuration, documentation, or unrelated code as part of this fix."
    ),
    "multi-swe-rl": (
        "The repository is checked out in your current working directory, at the state before "
        "this issue was fixed. Use your available tools to explore the codebase, understand the "
        "issue, and make the necessary changes directly using your edit/write tools — actually "
        "modify the files; do not just describe or show the fix in your response. Your final "
        "message does not need to include any code or diff — the changes you make to the files "
        "are the submission. Keep your change as small and targeted as possible — modify only "
        "what's strictly necessary to fix the described issue; do not update changelogs, CI "
        "configuration, documentation, or unrelated code as part of this fix."
    ),
}

# Used instead of _CONTEXT_INSTRUCTIONS when the model config says `supports_tool_calls: false`
# (currently: local llama.cpp providers). Confirmed empirically this session: given the
# tool-inviting instruction above, these models attempt a tool call in their own training-time
# dialect (e.g. `<function-calls>{...}</function-calls>`), but llama.cpp's OpenAI-compatible
# endpoint never translates that into `message.tool_calls` — Pi sees plain text, not a tool call,
# and returns the inert tool-call text as the "final answer", which always fails to apply as a
# patch. Asking directly for a diff — the same shape `_INSTRUCTIONS` already uses when there's no
# repo context at all — at least gets a gradeable answer instead of a guaranteed failure.
_CONTEXT_INSTRUCTIONS_NO_TOOLS = {
    "swe-smith": (
        "The repository is checked out in your current working directory, at the state before "
        "this issue was fixed, but you do NOT have working tool access in this environment — do "
        "not attempt to call any read/bash/edit/write tool. Based on the issue description alone, "
        "respond with a unified diff (git apply-compatible) that fixes it, and nothing else. Wrap "
        "it in a single ```diff code block."
    ),
    "swe-gym": (
        "The repository is checked out in your current working directory, at the state before "
        "this issue was fixed, but you do NOT have working tool access in this environment — do "
        "not attempt to call any read/bash/edit/write tool. Based on the issue description alone, "
        "respond with a unified diff (git apply-compatible) that fixes it, and nothing else. Wrap "
        "it in a single ```diff code block."
    ),
    "multi-swe-rl": (
        "The repository is checked out in your current working directory, at the state before "
        "this issue was fixed, but you do NOT have working tool access in this environment — do "
        "not attempt to call any read/bash/edit/write tool. Based on the issue description alone, "
        "respond with a unified diff (git apply-compatible) that fixes it, and nothing else. Wrap "
        "it in a single ```diff code block."
    ),
}


@dataclasses.dataclass(frozen=True)
class TokenUsage:
    """Pi's own reported usage/cost for one `pi -p` call, read from its `--mode json` event stream
    rather than computed from our own token estimate — Pi already prices every call against its
    built-in model catalog (confirmed: it reports accurate cost for claude-haiku-4-5 even though
    that model has no entry in our local ~/.pi/agent/models.json), so there's nothing for us to
    estimate."""
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclasses.dataclass(frozen=True)
class RunResult:
    solution: str | None  # None if the agent produced nothing usable — see grading/base.py's
    # error_no_solution outcome, which calibrate.py records directly without calling a grader.
    detail: str = ""
    raw_response: str = ""
    rate_limited: bool = False  # True routes calibrate.py to `error_harness` instead of
    # `error_no_solution` — a 429 is an infra/quota problem, not the model failing to answer, and
    # must not inflate its measured error rate.
    context_unavailable: bool = False  # True if repo_context setup (clone/checkout) itself failed,
    # before pi was ever invoked — an infra problem, not the model's fault, so calibrate.py must
    # route this to `error_harness` too, the same as rate_limited.
    harness_error: bool = False  # True when Pi's own event stream reports the PROVIDER rejected
    # the call (e.g. an API auth failure) — confirmed live this session that Pi exits 0 in this
    # case, so nothing else here would ever catch it. `detail` carries the provider's error message.
    # Routes to `error_harness`, same reasoning as rate_limited/context_unavailable: the call never
    # reached the model, so it can't be evidence of the model failing to answer.
    timed_out: bool = False  # True when OUR subprocess timeout fired (distinct from a grader's own
    # error_timeout, which is about the TEST run, not the model call). Previously fell through to
    # `error_no_solution` like a genuine empty response — but a call that never finished isn't
    # evidence the model couldn't solve the task, just that it didn't in the time we gave it.
    usage: TokenUsage | None = None  # None when pi's stdout wasn't parseable JSON (e.g. it never
    # ran, or emitted plain text) — a genuinely unknown cost, not a zero one; calibrate.py treats
    # that distinction as "0 measured" for the CSV since there's nothing else to report.


def build_prompt(task: Task, has_repo_context: bool = False, supports_tool_calls: bool = True) -> str:
    if has_repo_context:
        instructions = _CONTEXT_INSTRUCTIONS if supports_tool_calls else _CONTEXT_INSTRUCTIONS_NO_TOOLS
        context_instruction = instructions.get(task.source)
        if context_instruction is not None:
            return f"{task.prompt}\n\n{context_instruction}"
    instruction = _INSTRUCTIONS.get(task.source, "Respond with only the code that solves this.")
    return f"{task.prompt}\n\n{instruction.format(code_prompt=task.row.get('code_prompt', ''))}"


def extract_solution(response: str) -> str | None:
    match = _CODE_BLOCK_RE.search(response)
    if match:
        # `.strip("\n")` only — a bare `.strip()` here would also eat leading INDENTATION off the
        # first real line (it strips all leading whitespace, not just blank lines). That's fatal
        # for bigcodebench: the grader concatenates `code_prompt + solution` directly (code_prompt
        # ends mid-signature, e.g. "def task_func(...):\n"), so a solution missing its first
        # line's indentation always raises IndentationError regardless of whether the model's
        # logic was right. Confirmed empirically this session against real calibration output.
        extracted = match.group(1).strip("\n")
        return extracted or None
    stripped = response.strip()
    return stripped or None


def _parse_json_stream(stdout: str) -> tuple[str | None, TokenUsage | None, str | None]:
    """Parses `pi --mode json`'s newline-delimited event stream, returning the final assistant
    message's text, the call's total usage/cost, and — new — an API-level error message if the
    provider itself rejected the call.

    That third value matters because Pi's own process exit code does NOT reflect this: confirmed
    live this session that a real Anthropic 401 ("API key is invalid") left `pi` exiting 0, with
    the failure visible only as `stopReason: "error"` / `errorMessage: "..."` on the last assistant
    message. Before this, `run_pi` had no way to tell "the provider rejected the call" apart from
    "the model genuinely produced nothing" — both looked identical (empty text, zero usage), and
    calibrate.py counted the former as `error_no_solution` (graded, counts against the model)
    instead of `error_harness` (excluded) — a real, observed 15% of one model's calls in one run,
    all in the back half, consistent with a credential degrading partway through rather than being
    broken from the start.

    `agent_end` is always the last event and carries the full conversation, so it alone has
    everything needed — no need to track events as they stream by.

    Usage is summed across EVERY assistant message in the turn, not read off the last one alone —
    confirmed live this session that each assistant message's `usage` is PER-TURN, not cumulative,
    for a real tool-use conversation: a 6-turn repo-context call showed `input: 3` on every single
    turn and a small, DIFFERENT `cost.total` on each (e.g. $0.00035, $0.00013, ..., $0.00025) — only
    `totalTokens` (a separate, genuinely cumulative field we don't use) grows turn over turn. Taking
    only the last message, as an earlier version of this function did, silently kept just that
    final turn's cost and dropped the other 5 — a real ~7x undercount on that one call. A
    single-turn, no-tool-use call (the only shape this was originally verified against) has exactly
    one assistant message, where "sum across messages" and "last message" are the same number,
    which is why that earlier verification didn't catch this.

    Returns (None, None, None) on anything that isn't this NDJSON shape — e.g. a test's plain-text
    stdout fixture, or a real failure — so callers fall back to treating `stdout` as the raw
    response, exactly like before this format existed."""
    agent_end = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("type") == "agent_end":
            agent_end = event

    if agent_end is None:
        return None, None, None

    assistant_messages = [m for m in agent_end.get("messages", []) if m.get("role") == "assistant"]
    if not assistant_messages:
        return None, None, None

    last = assistant_messages[-1]
    text = "".join(
        block.get("text", "") for block in last.get("content", []) if block.get("type") == "text"
    )
    # Checked on the LAST message only: a mid-conversation error on an earlier turn that Pi
    # recovered from (retried and got a real response afterward) isn't a call-level failure — only
    # the call's own final state matters here, mirroring how `text` above is also read from `last`.
    error_message = last.get("errorMessage") if last.get("stopReason") == "error" else None

    total_input = 0
    total_output = 0
    total_cost = 0.0
    saw_usage = False
    for message in assistant_messages:
        usage_obj = message.get("usage")
        if not usage_obj:
            continue
        saw_usage = True
        total_input += usage_obj.get("input", 0)
        total_output += usage_obj.get("output", 0)
        total_cost += (usage_obj.get("cost") or {}).get("total", 0.0)

    usage = TokenUsage(input_tokens=total_input, output_tokens=total_output, cost_usd=total_cost) if saw_usage else None
    return text, usage, error_message


def run_pi(task: Task, model: ModelConfig, timeout_seconds: int, sleep=time.sleep) -> RunResult:
    if model.is_control:
        raise ValueError(f"run_pi called with a control model ({model.model_id}) — controls are synthesized, not run")

    cached_clone: Path | None = None
    worktree: Path | None = None
    remote_ref = repo_context.remote_and_ref(task)
    if remote_ref is not None:
        remote_url, ref = remote_ref
        try:
            cached_clone = repo_context.ensure_cached_clone(remote_url)
            worktree = repo_context.checkout_worktree(cached_clone, ref)
            repo_context.apply_post_checkout_setup(worktree, task)
        except repo_context.RepoContextError as e:
            logger.error(f"repo context unavailable for {task.task_id}: {e}")
            # apply_post_checkout_setup can fail AFTER checkout_worktree already created a real
            # worktree (unlike ensure_cached_clone/checkout_worktree, which clean up after
            # themselves on failure) — without this, that worktree would leak, since this early
            # return skips the try/finally below that normally owns worktree cleanup.
            if worktree is not None:
                repo_context.remove_worktree(cached_clone, worktree)
            return RunResult(solution=None, detail=str(e), context_unavailable=True)

    try:
        has_repo_context = worktree is not None
        prompt = build_prompt(task, has_repo_context=has_repo_context, supports_tool_calls=model.supports_tool_calls)
        args = [
            "pi", "-p", prompt,
            "--no-session",
            "--no-context-files",
            "--no-skills",
            "--no-prompt-templates",
            "--provider", model.provider,
            "--model", model.model_id,
            # NDJSON event stream instead of plain text — the only way to read back Pi's own
            # per-call usage/cost (see TokenUsage/_parse_json_stream). Solution extraction still
            # works exactly as before: _parse_json_stream falls back to (None, None) on anything
            # that isn't this shape, and extract_solution then runs on raw stdout same as always.
            "--mode", "json",
        ]
        # Pi offers its own built-in read/write/bash/edit tools to every model call by default —
        # confirmed empirically this session that this isn't harmless even for a model with
        # correctly-working tool calls (Llama-3.1-8B-Instruct): on a self-contained task with no
        # repo to act on (bigcodebench/ds1000), it can still attempt a tool call, and Pi's own
        # response parser rejects the malformed attempt outright ("The model produced output that
        # does not match the expected peg-native format", exit 1, before any real content comes
        # back — confirmed by reproducing the exact failing call with/without `--no-tools`).
        # Tools are only useful when BOTH a real worktree exists to act on AND the model is known
        # to translate its tool-call attempts into the wire protocol correctly — anything else and
        # offering them is pure downside.
        if not (has_repo_context and model.supports_tool_calls):
            args.append("--no-tools")

        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            _rate_limiter.wait(model.model_id, model.rate_limit_rpm)

            logger.info(f"pi call started: {model.model_id} on task {task.task_id} (timeout={timeout_seconds}s)")
            started = time.monotonic()
            try:
                proc = subprocess.run(
                    args, capture_output=True, text=True, timeout=timeout_seconds, check=False, cwd=worktree,
                )
            except subprocess.TimeoutExpired:
                logger.warning(f"pi call timed out after {timeout_seconds}s: {model.model_id} on task {task.task_id}")
                return RunResult(solution=None, detail=f"pi timed out after {timeout_seconds}s", timed_out=True)
            except FileNotFoundError:
                logger.error(f"pi binary not found on PATH ({model.model_id} on task {task.task_id})")
                return RunResult(solution=None, detail="pi binary not found on PATH")

            duration_s = round(time.monotonic() - started, 2)
            if proc.returncode != 0:
                rate_limited = _looks_rate_limited(proc.stderr)
                if rate_limited and attempt < MAX_RATE_LIMIT_RETRIES:
                    wait_s = _parse_retry_after_seconds(proc.stderr)
                    wait_s = min(wait_s if wait_s is not None else 10.0, MAX_RATE_LIMIT_WAIT_SECONDS) + 0.5
                    logger.warning(
                        f"rate limited after {duration_s}s, retrying in {wait_s:.1f}s "
                        f"(attempt {attempt + 1}/{MAX_RATE_LIMIT_RETRIES}): {model.model_id} on task {task.task_id}"
                    )
                    sleep(wait_s)
                    continue

                level = logger.warning if not rate_limited else logger.error
                level(
                    f"pi call failed (exit {proc.returncode}, rate_limited={rate_limited}) after {duration_s}s: "
                    f"{model.model_id} on task {task.task_id} — {proc.stderr[-300:].strip()}"
                )
                return RunResult(
                    solution=None,
                    detail=f"pi exit {proc.returncode}: {proc.stderr[-500:]}",
                    raw_response=proc.stdout,
                    rate_limited=rate_limited,
                )

            final_text, usage, error_message = _parse_json_stream(proc.stdout)
            if error_message is not None:
                # Pi itself exits 0 even when the underlying provider call failed (confirmed live:
                # a real Anthropic 401 "API key is invalid") — this is an infra/credential problem,
                # not the model failing to answer, so it must not be scored as error_no_solution.
                logger.error(
                    f"provider rejected the call (pi exited 0 but reported an error): "
                    f"{model.model_id} on task {task.task_id} — {error_message}"
                )
                return RunResult(solution=None, detail=error_message, raw_response=proc.stdout, harness_error=True)

            response_text = final_text if final_text is not None else proc.stdout

            # A tool-using agent's actual edits (captured via `git diff`) are preferred over
            # parsing its text response — only fall back to text extraction if the worktree came
            # back clean (no repo context, or the agent responded with prose instead of using its
            # tools).
            solution = repo_context.extract_diff(worktree) if worktree is not None else None
            if solution is None:
                solution = extract_solution(response_text)
            cost_note = f", cost=${usage.cost_usd:.5f}" if usage else ""
            logger.info(
                f"pi call completed in {duration_s}s: {model.model_id} on task {task.task_id} "
                f"(had_solution={solution is not None}{cost_note})"
            )
            return RunResult(solution=solution, raw_response=response_text, usage=usage)

        raise AssertionError("unreachable — the loop always returns on its last iteration")
    finally:
        if worktree is not None:
            repo_context.remove_worktree(cached_clone, worktree)
