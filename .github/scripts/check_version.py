"""Verify version strings match. Usage: check_version.py [expected_version]

Compares pyproject.toml, leech/__init__.py, and optional expected version
(e.g. a git tag name, leading "v" stripped). Exits 1 on mismatch.
"""
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent

pyproject = re.search(
    r'^version\s*=\s*"([^"]+)"', (root / "pyproject.toml").read_text(), re.M
).group(1)
init = re.search(
    r'__version__\s*=\s*"([^"]+)"', (root / "leech" / "__init__.py").read_text()
).group(1)

versions = {"pyproject.toml": pyproject, "leech/__init__.py": init}
if len(sys.argv) > 1:
    versions["expected"] = sys.argv[1].lstrip("vV")

print(" ".join(f"{k}={v}" for k, v in versions.items()))
if len(set(versions.values())) != 1:
    print(f"::error::Version mismatch: {versions}", file=sys.stderr)
    sys.exit(1)
