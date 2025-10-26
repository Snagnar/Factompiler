"""Diagnostic utilities used during semantic analysis."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from dsl_compiler.src.ast import ASTNode


class DiagnosticLevel(Enum):
    """Severity levels for compiler diagnostics."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class Diagnostic:
    """A compiler diagnostic message."""

    level: DiagnosticLevel
    message: str
    node: Optional[ASTNode] = None
    line: int = 0
    column: int = 0

    def __post_init__(self) -> None:
        if self.node:
            self.line = self.node.line
            self.column = self.node.column


class DiagnosticCollector:
    """Collects and manages diagnostic messages."""

    def __init__(self) -> None:
        self.diagnostics: List[Diagnostic] = []
        self.error_count = 0
        self.warning_count = 0

    def info(self, message: str, node: Optional[ASTNode] = None) -> None:
        """Add an informational diagnostic."""

        diag = Diagnostic(DiagnosticLevel.INFO, message, node)
        self.diagnostics.append(diag)

    def warning(self, message: str, node: Optional[ASTNode] = None) -> None:
        """Add a warning diagnostic."""

        diag = Diagnostic(DiagnosticLevel.WARNING, message, node)
        self.diagnostics.append(diag)
        self.warning_count += 1

    def error(self, message: str, node: Optional[ASTNode] = None) -> None:
        """Add an error diagnostic."""

        diag = Diagnostic(DiagnosticLevel.ERROR, message, node)
        self.diagnostics.append(diag)
        self.error_count += 1

    def has_errors(self) -> bool:
        """Check whether any errors were collected."""

        return self.error_count > 0

    def get_messages(self, level: Optional[DiagnosticLevel] = None) -> List[str]:
        """Return formatted diagnostic messages filtered by severity."""

        messages: List[str] = []
        for diag in self.diagnostics:
            if level is None or diag.level == level:
                prefix = diag.level.value.upper()
                location = f"{diag.line}:{diag.column}" if diag.line > 0 else "?"
                messages.append(f"{prefix} [{location}]: {diag.message}")
        return messages


class SemanticError(Exception):
    """Exception raised for semantic analysis errors."""

    def __init__(self, message: str, node: Optional[ASTNode] = None) -> None:
        self.message = message
        self.node = node
        location = f" at {node.line}:{node.column}" if node and node.line > 0 else ""
        super().__init__(f"{message}{location}")
