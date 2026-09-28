# Running `calibrate` unattended: pause, resume, tail logs

A full calibration run (or a single `--model` onboarding run) can take hours — worth detaching
from the terminal rather than running in the foreground. `nohup`+`disown` alone isn't enough to
safely pause it later: in a non-interactive shell, a backgrounded job's process group ID is **not**
the same as its own PID (job control isn't available), so the real PGID has to be captured
explicitly at launch time — otherwise pausing/resuming later targets the wrong thing, or nothing
at all.

## Launch

```bash
mkdir -p artifacts/logs
LOGFILE=artifacts/logs/calibrate-<label>-$(date -u +%Y%m%d-%H%M%S).log

nohup .venv/bin/router pipeline calibrate \
  --tasks-file artifacts/calibration-task-selection-<pin>.json \
  --model <model_id> \
  > "$LOGFILE" 2>&1 < /dev/null &

PID=$!
PGID=$(ps -o pgid= -p "$PID" | tr -d ' ')
echo "$PID"  > /tmp/calibrate-<label>.pid
echo "$PGID" > /tmp/calibrate-<label>.pgid
disown
```

## Controlling it

| Action | Command | Why |
|---|---|---|
| Tail logs | `tail -f "$LOGFILE"` | Captures the same stdout `typer.echo` progress lines a foreground run would show, redirected instead of printed. |
| Pause | `kill -STOP -$(cat /tmp/calibrate-<label>.pgid)` | Signals the whole **process group**, not just the Python PID — the `pi` subprocess mid-call inherits the same group, so it freezes too, not just the orchestrator waiting on it. |
| Resume | `kill -CONT -$(cat /tmp/calibrate-<label>.pgid)` | Picks up exactly where it froze — no lost work, no re-grading. |
| Check it's alive | `ps -p $(cat /tmp/calibrate-<label>.pid)` | Confirms it survived a closed terminal — `nohup` blocks the hangup signal that would otherwise kill it, `disown` removes it from the shell's own job table so shell exit can't touch it either. |

## There is no resume from a killed process

`calibrate_models` has no checkpoint/resume logic of its own — if the process is genuinely
terminated rather than paused (check with `ps -p <pid>`; a truly-gone process won't show up at
all, not even in a stopped `T` state), the only option is restarting the same command, which
re-grades every task from scratch and writes a fresh `calibration-details-<new-run>.csv`.

See also: [cli-reference.md](cli-reference.md) for every `calibrate` flag.
