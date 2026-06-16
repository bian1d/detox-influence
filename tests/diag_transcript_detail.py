"""Detailed look at all Bash commands across transcripts to find patterns for allowlist."""
import json
from pathlib import Path
from collections import Counter

TRANSCRIPT_DIR = Path.home() / ".claude" / "projects"

# What's already covered by ~/.claude/settings.json (user-level)
USER_LEVEL_COVERED = {
    "ls", "cat", "head", "tail", "wc", "grep", "egrep", "sed -n", "awk",
    "find", "file", "stat", "du", "df", "ps", "pgrep", "lsof",
    "nvidia-smi", "free", "uptime", "date", "pwd", "tree", "which", "whereis",
    "echo", "printenv", "env", "python", "python3", "python -c", "python3 -c",
    "python -m", "python3 -m", "torch.load", "jq", "diff", "md5sum", "sha256sum",
    "zcat", "unzip -l", "tar -tf", "tar tvf", "kill -0",
    # auto-allowed
    "grep", "egrep", "fgrep", "rg", "jq", "sort", "uniq", "tr", "cut", "sed",
    "diff", "wc", "find", "ls", "cat", "head", "tail", "stat", "which", "file",
    "tree", "df", "du", "free", "ps", "pgrep", "lsof", "date",
    # git/gh read-only
    "git status", "git log", "git diff", "git show", "git branch",
}

def main():
    files = sorted(
        TRANSCRIPT_DIR.rglob("*.jsonl"),
        key=lambda p: p.stat().st_mtime, reverse=True
    )[:50]

    all_cmds = []
    for fpath in files:
        try:
            for line in fpath.read_text(errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except:
                    continue
                if obj.get("type") == "assistant":
                    for content in obj.get("message", {}).get("content", []):
                        if isinstance(content, dict) and content.get("type") == "tool_use":
                            if content.get("name") == "Bash":
                                cmd = content.get("input", {}).get("command", "")
                                if cmd:
                                    all_cmds.append(cmd.strip())
        except:
            continue

    print(f"Total Bash calls found: {len(all_cmds)}")
    print(f"\nAll unique commands (truncated to 100 chars):")
    counter = Counter(all_cmds)
    for cmd, count in counter.most_common():
        print(f"  {count:>3}x  {cmd[:100]}")

if __name__ == "__main__":
    main()
