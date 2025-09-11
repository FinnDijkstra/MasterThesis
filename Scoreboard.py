import shutil
import math

# Optional Windows color support
try:
    import colorama; colorama.just_fix_windows_console()
except Exception:
    pass

# Colors
RESET = "\x1b[0m"
BOLD  = "\x1b[1m"
DIM   = "\x1b[2m"
COL = {
    "delivered": "\x1b[38;5;82m",    # green
    "random":    "\x1b[38;5;39m",    # blue
    "tie":       "\x1b[38;5;220m",   # yellow
    "bad":       "\x1b[38;5;196m",   # red
    "ok":        "\x1b[38;5;82m",
    "warn":      "\x1b[38;5;208m",
    "err":       "\x1b[38;5;196m",
    "title":     "\x1b[38;5;45m",
}

BLOCK = "█"
SEP   = "│"

def _term_width(default=80):
    try:
        return shutil.get_terminal_size((default, 20)).columns
    except Exception:
        return default

def _draw_bar(parts, total, width):
    """parts = [(name, count), ...] with names in COL; total >= 0"""
    if total <= 0:
        return DIM + "·" * width + RESET
    # ensure at least 1 cell for any nonzero part
    raw = [(name, (count / total) * width) for name, count in parts]
    # round nicely while preserving total width
    cells = [max(1 if x > 0 else 0, int(round(x))) for _, x in raw]
    # fix drift
    drift = width - sum(cells)
    i = 0
    while drift != 0 and len(cells) > 0:
        j = i % len(cells)
        if drift > 0:
            cells[j] += 1; drift -= 1
        elif drift < 0 and cells[j] > 0:
            cells[j] -= 1; drift += 1
        i += 1
    # build bar
    out = []
    for (name, _), c in zip(parts, cells):
        if c > 0:
            out.append(COL.get(name, "") + (BLOCK * c))
    return "".join(out) + RESET

def _legend(parts, total):
    cells = []
    for name, count in parts:
        pct = (100 * count / total) if total else 0.0
        cells.append(f"{COL.get(name,'')}{name}{RESET}:{count} ({pct:.1f}%)")
    return "  " + "  ".join(cells)

def _status_line(ok, msg):
    color = COL["ok"] if ok else (COL["warn"] if "mismatch" in msg else COL["err"])
    icon  = "✔" if ok else ("⚠" if "mismatch" in msg else "✖")
    return f"{color}{icon} {msg}{RESET}"

def show_dashboard(scoreboard, goalboard, facetsDone, goals):
    w = max(40, min(80, _term_width() - 6))
    print()
    print(f"{COL['title']}{BOLD}=== Scoreboard ==={RESET}")
    sb_parts = [(k, scoreboard.get(k, 0)) for k in ("delivered","random","tie","bad")]
    sb_total = sum(v for _, v in sb_parts)
    print(_draw_bar(sb_parts, sb_total, w))
    print(_legend(sb_parts, sb_total))
    ok_sb = (sb_total == facetsDone)
    diff_sb = sb_total - facetsDone
    msg_sb = f"score total = {sb_total}, facetsDone = {facetsDone}" + ("" if ok_sb else f" (mismatch {diff_sb:+d})")
    print(_status_line(ok_sb, msg_sb))

    print()
    print(f"{COL['title']}{BOLD}=== Goals ==={RESET}")
    gb_parts = [(k, goalboard.get(k, 0)) for k in ("delivered","random")]
    gb_total = sum(v for _, v in gb_parts)
    print(_draw_bar(gb_parts, gb_total, w))
    print(_legend(gb_parts, gb_total))
    ok_gb = (gb_total == goals)
    diff_gb = gb_total - goals
    msg_gb = f"goals total = {gb_total}, goals = {goals}" + ("" if ok_gb else f" (mismatch {diff_gb:+d})")
    print(_status_line(ok_gb, msg_gb))

    # a tiny “mood” meter for fun
    # delivered = scoreboard.get("delivered", 0)
    # bad = scoreboard.get("bad", 0)
    # mood = "😃" if delivered > 2*bad else ("🙂" if delivered >= bad else ("😐" if sb_total>0 else "🤔"))
    print()
    # print(f"{BOLD}Mood:{RESET} {mood}  {COL['delivered']}delivered{RESET}:{delivered}  {COL['bad']}bad{RESET}:{bad}")
    # print()

# --- Example ---
if __name__ == "__main__":
    scoreboard = {"delivered":7, "random":3, "tie":2, "bad":1}
    goalboard  = {"delivered":4, "random":1}
    facetsDone = 13
    goals      = 5
    show_dashboard(scoreboard, goalboard, facetsDone, goals)
