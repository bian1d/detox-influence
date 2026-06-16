"""Scan JSONL transcripts for Bash tool-call patterns to suggest allowlist entries."""
import json
import re
import os
from pathlib import Path
from collections import Counter

TRANSCRIPT_DIR = Path.home() / ".claude" / "projects"
MAX_FILES = 50

# Already auto-allowed by Claude Code — no point adding these
ALREADY_AUTO_ALLOWED = {
    "ls", "cat", "head", "tail", "wc", "stat", "find", "echo", "printf",
    "diff", "which", "file", "grep", "egrep", "fgrep", "rg", "jq", "tree",
    "df", "du", "free", "ps", "pgrep", "lsof", "date", "pwd", "whoami",
    "sed", "sort", "uniq", "tr", "cut", "basename", "dirname", "realpath",
    "sha256sum", "md5sum", "tput", "seq",
    # git read-only subcommands
    "git status", "git log", "git diff", "git show", "git blame",
    "git branch", "git tag", "git remote", "git ls-files", "git config",
    "git rev-parse", "git describe", "git stash", "git reflog",
    "git shortlog", "git cat-file", "git for-each-ref", "git worktree",
    # gh read-only
    "gh pr view", "gh pr list", "gh pr diff", "gh pr checks", "gh pr status",
    "gh issue view", "gh issue list", "gh run view", "gh run list",
    "gh workflow list", "gh repo view", "gh release view", "gh auth status",
    # docker read-only
    "docker ps", "docker images", "docker logs", "docker inspect",
}

# Commands that are NEVER safe to wildcard (arbitrary code execution risk)
DANGEROUS_WILDCARDS = {
    "python", "python3", "python3.10", "node", "bun", "deno", "ruby", "perl",
    "php", "lua", "bash", "sh", "zsh", "fish", "eval", "exec", "ssh",
    "npx", "bunx", "uvx", "uv",
    "npm", "yarn", "pnpm", "make", "just", "cargo", "go",
    "sudo", "docker run", "docker exec", "kubectl exec",
    "gh api",
}

# Read-only command tokens we DO want to allow
READONLY_SAFE = {
    "nvidia-smi", "torch", "pip", "conda",
    "tar", "unzip", "zcat", "env", "printenv",
    "uptime", "kill",  # kill -0 is read-only (check if process exists)
    "curl", "wget",    # only GET requests
    "jupyter", "ipython",
    "black", "mypy", "flake8", "pylint", "pytest",  # read-only analysis/test
}

def extract_leading_cmd(command: str) -> str | None:
    """Extract the leading command + first argument (e.g. 'nvidia-smi', 'tar -tf')."""
    # Strip env var prefixes like VAR=val cmd
    cmd = re.sub(r'^(?:[A-Z_][A-Z0-9_]*=\S+\s+)+', '', command.strip())
    # Strip timeout/time prefix
    cmd = re.sub(r'^(?:timeout|time)\s+\S+\s+', '', cmd)
    # Strip leading sudo
    cmd = re.sub(r'^sudo\s+', '', cmd)
    # Take the part before first pipe, &&, ||, ;
    cmd = re.split(r'[|;&]', cmd)[0].strip()
    tokens = cmd.split()
    if not tokens:
        return None
    base = tokens[0]
    # For known two-word commands (git, gh, docker), include subcommand
    if base in ("git", "gh", "docker", "kubectl") and len(tokens) > 1:
        return f"{base} {tokens[1]}"
    # For python/node etc — skip (dangerous wildcard)
    if base in DANGEROUS_WILDCARDS:
        return None
    return base

def is_readonly(cmd_key: str, raw_command: str) -> bool:
    """Heuristic: is this command read-only?"""
    # Anything modifying files
    mutating = {
        "rm", "mv", "cp", "mkdir", "touch", "chmod", "chown", "ln",
        "git add", "git commit", "git push", "git pull", "git merge",
        "git rebase", "git reset", "git checkout", "git switch",
        "git stash pop", "git stash drop", "git clean",
        "pip install", "pip uninstall", "conda install", "conda remove",
        "apt", "apt-get", "yum", "brew",
        "docker build", "docker run", "docker exec", "docker rm",
        "kubectl apply", "kubectl delete", "kubectl exec",
        "gh pr create", "gh pr merge", "gh pr edit",
        "gh issue create", "gh issue edit", "gh issue close",
        "wandb",
        "tee",   # writes to file
    }
    if cmd_key in mutating:
        return False
    # Check raw command for destructive flags
    if any(flag in raw_command for flag in [" -rf ", " --force", " -f ", ">/", ">>", " > "]):
        return False
    # Special cases
    if cmd_key == "kill" and "-0" not in raw_command:
        return False
    if cmd_key == "tar" and any(c in raw_command for c in ["x ", "-x", "extract"]):
        return False
    return True

def scan_transcripts():
    files = sorted(
        TRANSCRIPT_DIR.rglob("*.jsonl"),
        key=lambda p: p.stat().st_mtime, reverse=True
    )[:MAX_FILES]

    cmd_counter = Counter()
    raw_examples = {}

    for fpath in files:
        try:
            for line in fpath.read_text(errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Look for assistant tool_use messages
                if obj.get("type") == "assistant":
                    for content in obj.get("message", {}).get("content", []):
                        if isinstance(content, dict) and content.get("type") == "tool_use":
                            tool = content.get("name", "")
                            inp = content.get("input", {})
                            if tool == "Bash":
                                raw_cmd = inp.get("command", "")
                                cmd_key = extract_leading_cmd(raw_cmd)
                                if cmd_key:
                                    cmd_counter[cmd_key] += 1
                                    if cmd_key not in raw_examples:
                                        raw_examples[cmd_key] = raw_cmd[:120]
        except Exception:
            continue

    return cmd_counter, raw_examples

def main():
    print(f"Scanning transcripts in {TRANSCRIPT_DIR}...")
    cmd_counter, raw_examples = scan_transcripts()

    # Filter
    candidates = []
    for cmd_key, count in cmd_counter.most_common():
        if count < 3:
            continue
        # Skip already auto-allowed
        skip = False
        for auto in ALREADY_AUTO_ALLOWED:
            if cmd_key == auto or cmd_key.startswith(auto + " "):
                skip = True
                break
        if skip:
            continue
        # Skip dangerous wildcards
        base = cmd_key.split()[0]
        if base in DANGEROUS_WILDCARDS:
            continue
        # Skip non-readonly
        example = raw_examples.get(cmd_key, "")
        if not is_readonly(cmd_key, example):
            continue
        candidates.append((cmd_key, count, example))

    print(f"\nTop candidates (count >= 3, read-only, not auto-allowed):\n")
    print(f"{'#':>3}  {'Pattern':<35}  {'Count':>5}  Example")
    print("-" * 90)
    for i, (cmd_key, count, example) in enumerate(candidates[:30], 1):
        print(f"{i:>3}  {cmd_key:<35}  {count:>5}  {example[:50]}")

    print("\n\nAll raw counts (including auto-allowed and filtered):")
    for cmd, count in cmd_counter.most_common(50):
        print(f"  {count:>4}  {cmd}")

if __name__ == "__main__":
    main()
