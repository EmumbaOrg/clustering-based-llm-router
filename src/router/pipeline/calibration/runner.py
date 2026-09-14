"""Invokes the Pi coding agent headlessly (`pi -p`) to produce a candidate solution for a task
against a configured model — the only code path that calls a real agent (`reference`/`null`
controls are synthesized directly by calibrate.py instead). For sources `repo_context.py` can
check out, the solution is captured via `git diff` on a real worktree; otherwise it falls back to
extracting the first fenced code block from the agent's text response.
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
# Deliberately has no "summarize the change or include a diff" escape hatch — that option is a
# real bug, not just imprecise wording.
_CONTEXT_INSTRUCTIONS = {
    "swe-smith": (
        "The repository is checked out in your current working directory, at the state before "
        "this issue was fixed. Use your available tools to explore the codebase, understand the "
        "issue, and make the necessary changes directly using your edit/write tools — actually "
        "modify the files; do not just describe or show the fix in your response. Your final "
        "message does not need to include any code or diff — the changes you make to the files "
        "are the submission. Keep your change as small and targeted as possible — modify only "
        "what's strictly necessary to fix the described issue; do not update changelogs, CI "
        "configuration, documentation, test files, or unrelated code as part of this fix."
    ),
    "swe-gym": (
        "The repository is checked out in your current working directory, at the state before "
        "this issue was fixed. Use your available tools to explore the codebase, understand the "
        "issue, and make the necessary changes directly using your edit/write tools — actually "
        "modify the files; do not just describe or show the fix in your response. Your final "
        "message does not need to include any code or diff — the changes you make to the files "
        "are the submission. Keep your change as small and targeted as possible — modify only "
        "what's strictly necessary to fix the described issue; do not update changelogs, CI "
        "configuration, documentation, test files, or unrelated code as part of this fix."
    ),
    "multi-swe-rl": (
        "The repository is checked out in your current working directory, at the state before "
        "this issue was fixed. Use your available tools to explore the codebase, understand the "
        "issue, and make the necessary changes directly using your edit/write tools — actually "
        "modify the files; do not just describe or show the fix in your response. Your final "
        "message does not need to include any code or diff — the changes you make to the files "
        "are the submission. Keep your change as small and targeted as possible — modify only "
        "what's strictly necessary to fix the described issue; do not update changelogs, CI "
        "configuration, documentation, test files, or unrelated code as part of this fix."
    ),
}

# Used instead of _CONTEXT_INSTRUCTIONS when the model config says `supports_tool_calls: false`
# (local llama.cpp providers).
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
    rather than estimated locally — Pi's own number is treated as authoritative."""
    input_tokens: int
    output_tokens: int
    cost_usd: float
    turn_count: int  # number of assistant messages in agent_end's conversation (see
    # _parse_json_stream) — comes for free alongside the usage totals above.


@dataclasses.dataclass(frozen=True)
class RunResult:
    solution: str | None  # None if the agent produced nothing usable — see grading/base.py's
    # error_no_solution outcome, which calibrate.py records directly without calling a grader.
    detail: str = ""
    raw_response: str = ""
    context_unavailable: bool = False  # repo_context clone/checkout failed before pi ran — routes
    # to error_harness.
    harness_error: bool = False  # provider rejected the call (Pi itself still exits 0); `detail`
    # carries the provider's error message.
    timed_out: bool = False  # OUR subprocess timeout fired, distinct from a grader's own
    # error_timeout (the TEST run, not the model call).
    usage: TokenUsage | None = None  # None when pi's stdout wasn't parseable JSON — a genuinely
    # unknown cost, not a zero one; calibrate.py treats it as "0 measured" for the CSV.


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
        # `.strip("\n")` only — a bare `.strip()` would also eat leading INDENTATION off the first
        # real line, which is fatal for bigcodebench: the grader concatenates `code_prompt +
        # solution` directly (code_prompt ends mid-signature), so a de-indented first line always
        # raises IndentationError regardless of whether the model's logic was right.
        extracted = match.group(1).strip("\n")
        return extracted or None
    stripped = response.strip()
    return stripped or None


def _parse_json_stream(stdout: str) -> tuple[str | None, TokenUsage | None, str | None]:
    """Parses `pi --mode json`'s event stream: the final assistant message's text, total usage/cost
    (summed across every assistant message, not just the last), and an API-level error message if
    the provider rejected the call (Pi's own exit code doesn't reflect this). Returns
    (None, None, None) on anything that isn't this NDJSON shape, so callers fall back to treating
    `stdout` as the raw response."""
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

    usage = (
        TokenUsage(
            input_tokens=total_input, output_tokens=total_output, cost_usd=total_cost,
            turn_count=len(assistant_messages),
        )
        if saw_usage else None
    )
    return text, usage, error_message


def run_pi(task: Task, model: ModelConfig, timeout_seconds: int) -> RunResult:
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
            # NDJSON event stream — the only way to read back Pi's own per-call usage/cost.
            "--mode", "json",
        ]
        if not (has_repo_context and model.supports_tool_calls):
            args.append("--no-tools")

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
            logger.warning(
                f"pi call failed (exit {proc.returncode}) after {duration_s}s: "
                f"{model.model_id} on task {task.task_id} — {proc.stderr[-300:].strip()}"
            )
            # Pi exits 0 even when the provider/model itself fails, so a nonzero exit here means
            # Pi's own process failed (crash, OOM, disk full) — harness_error, not the model
            # producing nothing.
            return RunResult(
                solution=None, detail=f"pi exit {proc.returncode}: {proc.stderr[-500:]}", raw_response=proc.stdout,
                harness_error=True,
            )

        final_text, usage, error_message = _parse_json_stream(proc.stdout)
        if error_message is not None:
            # See RunResult.harness_error above for why this must not be scored as error_no_solution.
            logger.error(
                f"provider rejected the call (pi exited 0 but reported an error): "
                f"{model.model_id} on task {task.task_id} — {error_message}"
            )
            return RunResult(solution=None, detail=error_message, raw_response=proc.stdout, harness_error=True)

        response_text = final_text if final_text is not None else proc.stdout

        # A tool-using agent's actual edits (captured via `git diff`) are preferred over parsing
        # its text response — only fall back to text extraction if the worktree came back clean
        # (no repo context, or the agent responded with prose instead of using its tools).
        solution = repo_context.extract_diff(worktree) if worktree is not None else None
        if solution is None:
            solution = extract_solution(response_text)
        cost_note = f", cost=${usage.cost_usd:.5f}" if usage else ""
        logger.info(
            f"pi call completed in {duration_s}s: {model.model_id} on task {task.task_id} "
            f"(had_solution={solution is not None}{cost_note})"
        )
        return RunResult(solution=solution, raw_response=response_text, usage=usage)
    finally:
        if worktree is not None:
            repo_context.remove_worktree(cached_clone, worktree)
