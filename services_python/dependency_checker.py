"""Dependency safety checker for DistribAI job submissions.

Validates package requirements against:
- Known malicious packages (malware, spyware)
- Typosquatting detection (common misspellings of popular packages)
- System package blocking (os, sys, etc.)
- Dangerous package warnings (network packages that need review)
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from enum import Enum


class SafetyLevel(Enum):
    """Safety classification for dependencies."""

    SAFE = "safe"  # No issues
    WARNING = "warning"  # Needs review but allowed
    BLOCKED = "blocked"  # Hard block - cannot use


@dataclass
class DependencyCheck:
    """Result of a dependency safety check."""

    package_name: str
    version_spec: str | None
    safety_level: SafetyLevel
    reason: str
    suggestions: list[str]  # For typos, suggest correct packages
    requires_admin_approval: bool


# Known malicious/spyware packages - hard blocked
MALICIOUS_PACKAGES = {
    # Typosquats of popular packages
    "reqeusts",  # requests typo
    "urllib3-fake",  # fake urllib3
    "python-dateutil-fake",  # fake dateutil
    "pycrypto-fake",  # fake pycrypto
    "colorama-fake",  # fake colorama
    "certifi-fake",  # fake certifi
    "chardet-fake",  # fake chardet
    "idna-fake",  # fake idna
    "pygments-fake",  # fake pygments
    "six-fake",  # fake six
    "jinja2-fake",  # fake jinja2
    "markupsafe-fake",  # fake markupsafe
    "werkzeug-fake",  # fake werkzeug
    "click-fake",  # fake click
    "itsdangerous-fake",  # fake itsdangerous
    "flask-fake",  # fake flask
    "django-fake",  # fake django
    "numpy-fake",  # fake numpy
    "pandas-fake",  # fake pandas
    "scipy-fake",  # fake scipy
    "matplotlib-fake",  # fake matplotlib
    "pillow-fake",  # fake pillow
    "pyyaml-fake",  # fake pyyaml
    "requests-fake",  # fake requests
    "botocore-fake",  # fake botocore
    "s3transfer-fake",  # fake s3transfer
    "jmespath-fake",  # fake jmespath
    "dateutil-fake",  # fake python-dateutil
    # Malicious packages known to have been published
    "python3-dateutil",  # Confusion with python-dateutil
    "jeIlyfish",  # jellyfish typo (capital I)
    "crypt",  # Confusion with stdlib
    "pwd",  # Confusion with stdlib
    "socket",  # Confusion with stdlib
    "io-http",  # Confusion with aiohttp
    "yaml-fake",  # Fake yaml
    "pip-upgrader",  # Known malware
    "python-resources",  # Known malware
    "libpeshka",  # Known malware
    "apidev-coop",  # Known malware
    "mariab",  # Fake mariadb
    "pytoh",  # Fake python
    "urllib2",  # Confusion with urllib
    "sysconf",  # Known malware
    "mock-django",  # Known malware
}

# System packages that shouldn't be in requirements.txt
SYSTEM_PACKAGES = {
    "os",
    "sys",
    "time",
    "json",
    "re",
    "math",
    "random",
    "datetime",
    "collections",
    "itertools",
    "functools",
    "typing",
    "pathlib",
    "subprocess",
    "threading",
    "multiprocessing",
    "socket",
    "builtins",
    "__future__",
    "abc",
    "enum",
    "dataclasses",
    "contextlib",
    "inspect",
    "types",
    "warnings",
    "traceback",
    "copy",
    "pickle",
    "marshal",
    "io",
    "string",
    "hashlib",
    "base64",
    "binascii",
    "struct",
    "tempfile",
    "shutil",
    "glob",
    "fnmatch",
    "linecache",
    "textwrap",
}

# Dangerous packages that need admin approval
DANGEROUS_PACKAGES = {
    # Network/HTTP clients
    "requests",
    "urllib3",
    "http.client",
    "httpx",
    "aiohttp",
    "httplib2",
    "pycurl",
    "wget",
    "curl",
    # Database connectors
    "psycopg2",
    "pymysql",
    "sqlite3",
    "sqlalchemy",
    "pymongo",
    "redis",
    "elasticsearch",
    "cassandra-driver",
    # System access
    "paramiko",
    "fabric",
    "ansible",
    "salt",
    "puppet",
    "docker",
    "kubernetes",
    "boto3",
    "azure",
    "google-cloud",
    # Crypto/Security (legitimate but dangerous if misused)
    "pycrypto",
    "cryptography",
    "pyopenssl",
    "pynacl",
    "bcrypt",
    "argon2",
    "scrypt",
    "hashlib",
    # Code execution
    "eval",
    "exec",
    "compile",
    "importlib",
    "runpy",
    # File system
    "send2trash",
    "shutil",
    "pathlib",
    "os",
    "sys",
    # Process management
    "psutil",
    "subprocess",
    "multiprocessing",
    "signal",
    # Serialization (dangerous if untrusted)
    "pickle",
    "marshal",
    "shelve",
    "dill",
    "cloudpickle",
    # Remote execution
    "rpyc",
    "pyro4",
    "pyro5",
    "zerorpc",
    "celery",
}

# Popular packages for typosquat detection
POPULAR_PACKAGES = {
    "requests",
    "urllib3",
    "certifi",
    "charset-normalizer",
    "idna",
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "seaborn",
    "plotly",
    "scikit-learn",
    "tensorflow",
    "torch",
    "torchvision",
    "transformers",
    "datasets",
    "tokenizers",
    "accelerate",
    "peft",
    "bitsandbytes",
    "flask",
    "django",
    "fastapi",
    "starlette",
    "uvicorn",
    "gunicorn",
    "sqlalchemy",
    "alembic",
    "psycopg2-binary",
    "pymongo",
    "redis",
    "boto3",
    "botocore",
    "s3transfer",
    "jmespath",
    "python-dateutil",
    "pyyaml",
    "toml",
    "json5",
    "protobuf",
    "grpcio",
    "pillow",
    "opencv-python",
    "tqdm",
    "colorama",
    "rich",
    "pytest",
    "pytest-cov",
    "black",
    "isort",
    "flake8",
    "mypy",
    "jupyter",
    "ipython",
    "notebook",
    "ipywidgets",
    "aiohttp",
    "asyncio",
    "aiofiles",
    "aiobotocore",
    "cryptography",
    "pyjwt",
    "bcrypt",
    "passlib",
    "pydantic",
    "httpx",
    "websockets",
}


class DependencyChecker:
    """Checks package dependencies for safety issues."""

    def __init__(self):
        self.malicious = MALICIOUS_PACKAGES
        self.system = SYSTEM_PACKAGES
        self.dangerous = DANGEROUS_PACKAGES
        self.popular = POPULAR_PACKAGES

    def check_dependency(self, requirement: str) -> DependencyCheck:
        """Check a single dependency for safety issues.

        Args:
            requirement: Package requirement string (e.g., "requests>=2.28.0")

        Returns:
            DependencyCheck with safety classification
        """
        # Parse package name and version
        package_name, version_spec = self._parse_requirement(requirement)
        normalized_name = package_name.lower().strip()

        # Check 1: Malicious packages - hard block
        if normalized_name in self.malicious:
            suggestions = self._get_typo_suggestions(normalized_name)
            return DependencyCheck(
                package_name=package_name,
                version_spec=version_spec,
                safety_level=SafetyLevel.BLOCKED,
                reason=f"'{package_name}' is a known malicious or typosquat package. It may contain malware or spyware.",
                suggestions=suggestions,
                requires_admin_approval=False,  # Cannot be overridden
            )

        # Check 2: System packages
        if normalized_name in self.system:
            return DependencyCheck(
                package_name=package_name,
                version_spec=version_spec,
                safety_level=SafetyLevel.BLOCKED,
                reason=f"'{package_name}' is a Python standard library module. It doesn't need to be installed via pip.",
                suggestions=[],
                requires_admin_approval=False,
            )

        # Check 3: Typosquatting detection
        if normalized_name not in self.popular:
            closest_match = self._find_closest_popular(normalized_name)
            if closest_match and self._is_likely_typo(normalized_name, closest_match):
                return DependencyCheck(
                    package_name=package_name,
                    version_spec=version_spec,
                    safety_level=SafetyLevel.BLOCKED,
                    reason=f"'{package_name}' may be a typosquat of popular package '{closest_match}'. Typosquats can contain malware.",
                    suggestions=[closest_match],
                    requires_admin_approval=True,  # Admin can override with warnings
                )

        # Check 4: Dangerous packages needing review
        if normalized_name in self.dangerous:
            return DependencyCheck(
                package_name=package_name,
                version_spec=version_spec,
                safety_level=SafetyLevel.WARNING,
                reason=f"'{package_name}' provides network/system access capabilities. This requires admin approval.",
                suggestions=[],
                requires_admin_approval=True,
            )

        # Safe package
        return DependencyCheck(
            package_name=package_name,
            version_spec=version_spec,
            safety_level=SafetyLevel.SAFE,
            reason="No safety issues detected.",
            suggestions=[],
            requires_admin_approval=False,
        )

    def check_requirements(self, requirements: list[str]) -> list[DependencyCheck]:
        """Check a list of requirements.

        Args:
            requirements: List of requirement strings

        Returns:
            List of DependencyCheck results
        """
        return [self.check_dependency(req) for req in requirements if req.strip()]

    def has_blocked_packages(self, checks: list[DependencyCheck]) -> bool:
        """Check if any package is blocked."""
        return any(c.safety_level == SafetyLevel.BLOCKED for c in checks)

    def needs_admin_approval(self, checks: list[DependencyCheck]) -> bool:
        """Check if any package needs admin approval."""
        return any(c.requires_admin_approval for c in checks)

    def get_blockers(self, checks: list[DependencyCheck]) -> list[DependencyCheck]:
        """Get all blocked packages."""
        return [c for c in checks if c.safety_level == SafetyLevel.BLOCKED]

    def get_warnings(self, checks: list[DependencyCheck]) -> list[DependencyCheck]:
        """Get all warning-level packages."""
        return [c for c in checks if c.safety_level == SafetyLevel.WARNING]

    def get_safe_packages(self, checks: list[DependencyCheck]) -> list[DependencyCheck]:
        """Get all safe packages."""
        return [c for c in checks if c.safety_level == SafetyLevel.SAFE]

    def _parse_requirement(self, requirement: str) -> tuple[str, str | None]:
        """Parse a requirement string into package name and version spec.

        Examples:
            "requests>=2.28.0" -> ("requests", ">=2.28.0")
            "numpy==1.24.0" -> ("numpy", "==1.24.0")
            "torch" -> ("torch", None)
        """
        req = requirement.strip()

        # Find version specifier
        for sep in ["==", ">=", "<=", ">", "<", "~=", "!=", "===", ";"]:
            if sep in req:
                parts = req.split(sep, 1)
                # Handle extras (e.g., "requests[security]")
                name = parts[0].strip()
                version = sep + parts[1].strip() if len(parts) > 1 else None
                return name, version

        return req, None

    def _get_typo_suggestions(self, package_name: str) -> list[str]:
        """Get suggestions for a potentially misspelled package name."""
        normalized = package_name.lower()

        # Direct matches for common typos
        typo_map = {
            "reqeusts": "requests",
            "reqests": "requests",
            "requsets": "requests",
            "reqeust": "requests",
            "urllib": "urllib3",
            "urlib3": "urllib3",
            "urlib": "urllib3",
            "numPy": "numpy",
            "nunpy": "numpy",
            "numpi": "numpy",
            "pandas-python": "pandas",
            "pandass": "pandas",
            "matplot": "matplotlib",
            "matplot-lib": "matplotlib",
            "pytorch": "torch",
            "py-torch": "torch",
            "transformer": "transformers",
            "huggingface": "transformers",
            "hugging-face": "transformers",
            "sklearn": "scikit-learn",
            "scikitlearn": "scikit-learn",
            "sci-kit-learn": "scikit-learn",
            "flask-app": "flask",
            "djnago": "django",
            "djano": "django",
            "fast-api": "fastapi",
            "sql-alchemy": "sqlalchemy",
            "psycopg": "psycopg2-binary",
            "pillow-python": "pillow",
            "PIL": "pillow",
            "yaml": "pyyaml",
            "python-yaml": "pyyaml",
            "dateutil": "python-dateutil",
        }

        if normalized in typo_map:
            return [typo_map[normalized]]

        # Use fuzzy matching for other cases
        return self._find_closest_matches(package_name)

    def _find_closest_matches(self, package_name: str, n: int = 3) -> list[str]:
        """Find closest matching popular packages using fuzzy matching."""
        matches = difflib.get_close_matches(
            package_name.lower(),
            self.popular,
            n=n,
            cutoff=0.6,
        )
        return list(matches)

    def _find_closest_popular(self, package_name: str) -> str | None:
        """Find the single closest popular package."""
        matches = difflib.get_close_matches(
            package_name.lower(),
            self.popular,
            n=1,
            cutoff=0.7,  # Higher threshold for typosquat detection
        )
        return matches[0] if matches else None

    def _is_likely_typo(self, name1: str, name2: str) -> bool:
        """Determine if two names are likely a typo relationship.

        Uses Levenshtein distance ratio to detect typosquats.
        """
        # Normalize both names
        n1 = name1.lower().replace("-", "").replace("_", "")
        n2 = name2.lower().replace("-", "").replace("_", "")

        # Calculate similarity
        similarity = difflib.SequenceMatcher(None, n1, n2).ratio()

        # Common typo patterns:
        # - Single character substitution (requests -> reqeusts)
        # - Missing character (urllib3 -> urlib3)
        # - Extra character (pandas -> pandass)
        # - Swapped characters (numpy -> nunpy)

        # If very similar but not identical
        if similarity > 0.85 and similarity < 1.0:
            return True

        # Check for known typo patterns
        if len(n1) == len(n2):
            diff_count = sum(1 for a, b in zip(n1, n2, strict=True) if a != b)
            if diff_count <= 1:  # Single character difference
                return True

        # Length difference of 1 (missing/extra character)
        if abs(len(n1) - len(n2)) == 1:
            shorter, longer = (n1, n2) if len(n1) < len(n2) else (n2, n1)
            # Check if shorter is substring of longer with one extra char
            for i in range(len(longer)):
                test = longer[:i] + longer[i + 1 :]
                if test == shorter:
                    return True

        return False


# Global instance
_checker: DependencyChecker | None = None


def get_dependency_checker() -> DependencyChecker:
    """Get or create global dependency checker."""
    global _checker
    if _checker is None:
        _checker = DependencyChecker()
    return _checker


def quick_check(requirement: str) -> DependencyCheck:
    """Quick check a single requirement."""
    return get_dependency_checker().check_dependency(requirement)


def check_requirements_list(requirements: list[str]) -> dict:
    """Check a list of requirements and return summary.

    Returns dict with:
        - safe: list of safe packages
        - warnings: list of warning-level packages
        - blocked: list of blocked packages
        - needs_approval: whether admin approval is required
    """
    checker = get_dependency_checker()
    checks = checker.check_requirements(requirements)

    return {
        "safe": checker.get_safe_packages(checks),
        "warnings": checker.get_warnings(checks),
        "blocked": checker.get_blockers(checks),
        "needs_approval": checker.needs_admin_approval(checks),
        "can_proceed": not checker.has_blocked_packages(checks),
        "all_checks": checks,
    }
