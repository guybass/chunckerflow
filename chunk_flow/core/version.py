"""Version tracking and compatibility checking."""

from typing import Tuple


class Version:
    """Semantic version representation and comparison."""

    def __init__(self, version_string: str) -> None:
        """
        Initialize version from string.

        Args:
            version_string: Semantic version (e.g., "1.0.0", "2.1.0-alpha")
        """
        self.original = version_string
        parts = version_string.split("-")[0]  # Remove suffix (alpha, beta, etc.)
        self.major, self.minor, self.patch = map(int, parts.split("."))

    def __str__(self) -> str:
        """Return string representation."""
        return self.original

    def __repr__(self) -> str:
        """Return detailed representation."""
        return f"Version('{self.original}')"

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, Version):
            return NotImplemented
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)

    def __lt__(self, other: "Version") -> bool:
        """Check if this version is less than other."""
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __le__(self, other: "Version") -> bool:
        """Check if this version is less than or equal to other."""
        return self == other or self < other

    def __gt__(self, other: "Version") -> bool:
        """Check if this version is greater than other."""
        return not self <= other

    def __ge__(self, other: "Version") -> bool:
        """Check if this version is greater than or equal to other."""
        return not self < other

    def as_tuple(self) -> Tuple[int, int, int]:
        """Return version as tuple."""
        return (self.major, self.minor, self.patch)

    def is_compatible_with(self, other: "Version") -> bool:
        """
        Check if this version is compatible with another.

        Compatibility rules (SemVer):
        - Major version must match (breaking changes)
        - Minor/patch versions can differ

        Args:
            other: Version to check compatibility with

        Returns:
            True if compatible, False otherwise
        """
        return self.major == other.major


class Versioned:
    """Mixin for version tracking in components."""

    VERSION = "1.0.0"

    @classmethod
    def get_version(cls) -> Version:
        """Get component version."""
        return Version(cls.VERSION)

    @classmethod
    def is_compatible_with(cls, version_string: str) -> bool:
        """
        Check if this component is compatible with given version.

        Args:
            version_string: Version to check compatibility with

        Returns:
            True if compatible, False otherwise
        """
        other = Version(version_string)
        current = cls.get_version()
        return current.is_compatible_with(other)
