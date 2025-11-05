from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Any
from pathlib import Path

"""Unified diagnostic collection for the entire compiler pipeline."""


class DiagnosticSeverity(Enum):
    """Severity levels for compiler diagnostics."""

    DEBUG = "debug"  # Internal compiler information
    INFO = "info"  # Informational messages for users
    WARNING = "warning"  # Issues that don't prevent compilation
    ERROR = "error"  # Issues that prevent successful compilation


@dataclass
class Diagnostic:
    """A single diagnostic message with context."""

    severity: DiagnosticSeverity
    message: str
    stage: str  # parsing, semantic, lowering, layout, emission
    line: int = 0
    column: int = 0
    source_file: Optional[str] = None
    node: Optional[Any] = None  # ASTNode reference if available


class ProgramDiagnostics:
    """Central diagnostic collection for the entire compilation process.

    This class tracks all diagnostics across all compiler stages and provides
    methods for querying, filtering, and formatting them for user output.

    Usage:
        diagnostics = ProgramDiagnostics()
        diagnostics.error("Undefined variable", stage="semantic", line=10)
        if diagnostics.has_errors():
            print(diagnostics.format_for_user())
    """

    def __init__(
        self, verbose: bool = False, debug: bool = False, explain: bool = False
    ):
        self.diagnostics: List[Diagnostic] = []
        self.verbose = verbose
        self.debug = debug
        self.explain = explain
        self._error_count = 0
        self._warning_count = 0
        self.default_stage = "unknown"

    def info(
        self,
        message: str,
        stage: str | None = None,
        line: int = 0,
        column: int = 0,
        source_file: Optional[str] = None,
        node: Optional[Any] = None,
    ) -> None:
        """Add an informational message (shown in verbose mode)."""
        if self.verbose:
            self._add(
                DiagnosticSeverity.INFO, message, stage, line, column, source_file, node
            )

    def warning(
        self,
        message: str,
        stage: str | None = None,
        line: int = 0,
        column: int = 0,
        source_file: Optional[str] = None,
        node: Optional[Any] = None,
    ) -> None:
        """Add a warning (always shown, doesn't stop compilation)."""
        self._add(
            DiagnosticSeverity.WARNING, message, stage, line, column, source_file, node
        )
        self._warning_count += 1

    def error(
        self,
        message: str,
        stage: str | None = None,
        line: int = 0,
        column: int = 0,
        source_file: Optional[str] = None,
        node: Optional[Any] = None,
    ) -> None:
        """Add an error (always shown, stops compilation)."""
        self._add(
            DiagnosticSeverity.ERROR, message, stage, line, column, source_file, node
        )
        self._error_count += 1

    def _add(
        self,
        severity: DiagnosticSeverity,
        message: str,
        stage: str | None,
        line: int,
        column: int,
        source_file: Optional[str],
        node: Optional[Any],
    ) -> None:
        """Internal method to add a diagnostic."""
        # Extract location from node if provided and location not specified
        if node is not None and line == 0:
            line = getattr(node, "line", 0)
            column = getattr(node, "column", 0)
            if source_file is None:
                source_file = getattr(node, "source_file", None)

        diag = Diagnostic(
            severity=severity,
            message=message,
            stage=stage or self.default_stage,
            line=line,
            column=column,
            source_file=source_file,
            node=node,
        )
        self.diagnostics.append(diag)

    def has_errors(self) -> bool:
        """Check if any errors have been recorded."""
        return self._error_count > 0

    def error_count(self) -> int:
        """Get the number of errors."""
        return self._error_count

    def warning_count(self) -> int:
        """Get the number of warnings."""
        return self._warning_count

    def get_messages(
        self, min_severity: DiagnosticSeverity = DiagnosticSeverity.WARNING
    ) -> List[str]:
        """Get formatted messages at or above the specified severity level."""
        messages = []
        for diag in self.diagnostics:
            # Filter by severity
            severity_order = [
                DiagnosticSeverity.DEBUG,
                DiagnosticSeverity.INFO,
                DiagnosticSeverity.WARNING,
                DiagnosticSeverity.ERROR,
            ]
            if severity_order.index(diag.severity) < severity_order.index(min_severity):
                continue

            messages.append(self._format_diagnostic(diag))
        return messages

    def _format_diagnostic(self, diag: Diagnostic) -> str:
        """Format a single diagnostic for display."""
        # Format: SEVERITY [stage:file:line:col]: message
        parts = [diag.severity.value.upper()]

        location_parts = [diag.stage]
        if diag.source_file:
            location_parts.append(Path(diag.source_file).name)
        if diag.line > 0:
            location_parts.append(str(diag.line))
            if diag.column > 0:
                location_parts.append(str(diag.column))

        location = ":".join(location_parts)
        return f"{parts[0]} [{location}]: {diag.message}"

    def format_for_user(self) -> str:
        """Format all diagnostics for user-friendly output."""
        if not self.diagnostics:
            return "No diagnostics."

        # Get messages at appropriate level
        if self.debug:
            min_severity = DiagnosticSeverity.DEBUG
        elif self.verbose:
            min_severity = DiagnosticSeverity.INFO
        else:
            min_severity = DiagnosticSeverity.WARNING

        messages = self.get_messages(min_severity)

        # Add summary
        summary = f"\nCompilation summary: {self._error_count} error(s), {self._warning_count} warning(s)"

        return "\n".join(messages) + summary

    def merge(self, other: "ProgramDiagnostics") -> None:
        """Merge diagnostics from another collector."""
        self.diagnostics.extend(other.diagnostics)
        self._error_count += other._error_count
        self._warning_count += other._warning_count
