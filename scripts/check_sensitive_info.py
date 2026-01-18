#!/usr/bin/env python3
"""
Sensitive Information Checker - for Git pre-commit hook

Detects sensitive information in staged files to prevent accidental commits.
"""

import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


# 敏感信息检测规则
SENSITIVE_PATTERNS = [
    # Real passwords (excluding placeholders and public test accounts)
    (
        r'["\']密码["\']\s*:\s*["\'](?!your_password|123456|password|<your_|\.\.\.)[^"\']{6,}["\']',
        "Possible real password detected"
    ),
    (
        r'["\']password["\']\s*:\s*["\'](?!your_password|123456|password|<your_|\.\.\.)[^"\']{6,}["\']',
        "Possible real password detected"
    ),

    # Token/API Key (at least 32 random characters)
    (
        r'(?:token|api[_-]?key|secret[_-]?key|access[_-]?key)["\']?\s*[:=]\s*["\']?(?!your_|<your_|\.\.\.)([a-zA-Z0-9_-]{32,})',
        "Possible Token/API Key detected"
    ),

    # Private key
    (
        r'-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----',
        "Private key detected"
    ),

    # Database connection string (with real password)
    (
        r'(?:mysql|postgresql|mongodb)://[^:]+:(?!password|123456)[^@]{6,}@',
        "Database connection string detected (may contain real password)"
    ),
]

# File path patterns that should NOT be committed
FORBIDDEN_FILES = [
    r'\.vntrader/connect_[a-z]+\.json$',  # CTP/TTS config files
    r'\.vntrader/.*\.json$',               # Other .vntrader configs
    r'.*_secret\..*',                      # Files containing _secret
    r'.*\.pem$',                           # Private key files
    r'.*\.key$',                           # Key files
]

# Allowed placeholders (not considered sensitive)
ALLOWED_PLACEHOLDERS = [
    'your_username',
    'your_password',
    'your_token',
    'your_simnow_account',
    'your_simnow_password',
    '<your_',
    '...',
]


def get_staged_files() -> List[str]:
    """Get list of files in git staging area"""
    try:
        result = subprocess.run(
            ['git', 'diff', '--cached', '--name-only', '--diff-filter=ACM'],
            capture_output=True,
            text=True,
            check=True
        )
        files = [f.strip() for f in result.stdout.split('\n') if f.strip()]
        return files
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to get staged files: {e}", file=sys.stderr)
        return []


def check_forbidden_files(files: List[str]) -> List[Tuple[str, str]]:
    """Check if any files should not be committed"""
    violations = []
    for file_path in files:
        for pattern in FORBIDDEN_FILES:
            if re.search(pattern, file_path):
                violations.append((file_path, f"不应提交的文件类型（匹配规则: {pattern}）"))
                break
    return violations


def check_file_content(file_path: str) -> List[Tuple[int, str, str]]:
    """Check if file content contains sensitive information

    Returns:
        List of (line_number, line_content, reason)
    """
    violations = []

    try:
        # Read file content
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # Check line by line
        for line_num, line in enumerate(lines, start=1):
            # Skip comment lines
            if line.strip().startswith('#') or line.strip().startswith('//'):
                continue

            # Check each sensitive pattern
            for pattern, reason in SENSITIVE_PATTERNS:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    # Check if it's an allowed placeholder
                    matched_text = match.group(0)
                    is_placeholder = any(placeholder in matched_text for placeholder in ALLOWED_PLACEHOLDERS)

                    if not is_placeholder:
                        violations.append((line_num, line.strip(), reason))

    except Exception as e:
        # Skip if binary file or cannot read
        pass

    return violations


def main() -> int:
    """Main function"""
    print("[SECURITY] Checking for sensitive information...")
    print()

    # Get staged files
    staged_files = get_staged_files()

    if not staged_files:
        print("[OK] No staged files")
        return 0

    has_violations = False

    # 1. Check forbidden files
    forbidden_violations = check_forbidden_files(staged_files)
    if forbidden_violations:
        has_violations = True
        print("[ERROR] Found forbidden files:")
        for file_path, reason in forbidden_violations:
            print(f"   [FILE] {file_path}")
            print(f"          Reason: {reason}")
        print()

    # 2. Check file content for sensitive information
    content_violations = []
    for file_path in staged_files:
        # Skip binary files
        if any(file_path.endswith(ext) for ext in ['.bo', '.fe', '.db', '.pyc', '.so', '.dll']):
            continue

        violations = check_file_content(file_path)
        if violations:
            content_violations.append((file_path, violations))

    if content_violations:
        has_violations = True
        print("[ERROR] Found sensitive information in files:")
        for file_path, violations in content_violations:
            print(f"   [FILE] {file_path}:")
            for line_num, line, reason in violations[:3]:  # Show first 3
                print(f"          Line {line_num}: {reason}")
                print(f"             {line[:80]}...")
            if len(violations) > 3:
                print(f"          ... and {len(violations) - 3} more issues")
        print()

    # 3. Output results
    if has_violations:
        print("=" * 60)
        print("[FAIL] Sensitive information check failed, commit blocked")
        print()
        print("How to fix:")
        print("1. Remove files with sensitive info from staging:")
        print("   git restore --staged <file>")
        print("2. Replace sensitive info with placeholders:")
        print("   Use 'your_username', 'your_password', etc.")
        print("3. Ensure sensitive config files are in .gitignore")
        print()
        print("To force commit (NOT recommended):")
        print("   git commit --no-verify")
        print("=" * 60)
        return 1
    else:
        print("[PASS] Sensitive information check passed")
        return 0


if __name__ == '__main__':
    sys.exit(main())
