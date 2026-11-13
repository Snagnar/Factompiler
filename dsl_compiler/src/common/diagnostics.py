import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

"""Unified diagnostic collection for the entire compiler pipeline."""


class CompilationError(RuntimeError):
    """Raised for user-facing compile errors when ``raise_errors`` is enabled.

    It is a ``RuntimeError`` subclass for backwards compatibility. The CLI
    catches it and prints the message without a Python traceback.
    """

    def __init__(self, message: str, diagnostic: "Diagnostic | None" = None):
        super().__init__(message)
        self.diagnostic = diagnostic


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
    source_file: str | None = None
    node: Any | None = None  # ASTNode reference if available


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

    def __init__(self, log_level: str = "info", raise_errors: bool = False):
        self.diagnostics: list[Diagnostic] = []
        self.log_level = log_level
        self.raise_errors = raise_errors
        self._error_count = 0
        self._warning_count = 0
        self.default_stage = "unknown"

    def info(
        self,
        message: str,
        stage: str | None = None,
        line: int = 0,
        column: int = 0,
        source_file: str | None = None,
        node: Any | None = None,
    ) -> None:
        """Add an informational message (shown in verbose mode)."""
        diag = self._add(DiagnosticSeverity.INFO, message, stage, line, column, source_file, node)
        logging.info(self._log_text(diag))

    def warning(
        self,
        message: str,
        stage: str | None = None,
        line: int = 0,
        column: int = 0,
        source_file: str | None = None,
        node: Any | None = None,
    ) -> None:
        """Add a warning (always shown, doesn't stop compilation)."""
        diag = self._add(
            DiagnosticSeverity.WARNING, message, stage, line, column, source_file, node
        )
        logging.warning(self._log_text(diag))
        self._warning_count += 1

    def error(
        self,
        message: str,
        stage: str | None = None,
        line: int = 0,
        column: int = 0,
        source_file: str | None = None,
        node: Any | None = None,
    ) -> None:
        """Add an error (always shown, stops compilation)."""
        diag = self._add(DiagnosticSeverity.ERROR, message, stage, line, column, source_file, node)
        logging.error(self._log_text(diag))
        if self.raise_errors:
            raise CompilationError(self._log_text(diag), diag)
        self._error_count += 1

    def _add(
        self,
        severity: DiagnosticSeverity,
        message: str,
        stage: str | None,
        line: int,
        column: int,
        source_file: str | None,
        node: Any | None,
    ) -> Diagnostic:
        """Internal method to add a diagnostic."""
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
        return diag

    @staticmethod
    def _location(diag: Diagnostic) -> str:
        """``file.facto:line:col`` (parts omitted when unknown)."""
        parts = []
        if diag.source_file and diag.source_file != "<string>":
            parts.append(Path(diag.source_file).name)
        if diag.line > 0:
            parts.append(str(diag.line))
            if diag.column > 0:
                parts.append(str(diag.column))
        return ":".join(parts)

    def _log_text(self, diag: Diagnostic) -> str:
        """Text used for log output: ``[stage] file.facto:3:7: message``."""
        location = self._location(diag)
        prefix = f"[{diag.stage}] "
        return f"{prefix}{location}: {diag.message}" if location else prefix + diag.message

    def has_errors(self) -> bool:
        """Check if any errors have been recorded."""
        return self._error_count > 0

    def error_count(self) -> int:
        """Return the number of errors recorded."""
        return self._error_count

    def warning_count(self) -> int:
        """Return the number of warnings recorded."""
        return self._warning_count

    def _format_diagnostic(self, diag: Diagnostic) -> str:
        """Format a single diagnostic for display: ``ERROR: [stage] file:3:7: msg``."""
        return f"{diag.severity.value.upper()}: {self._log_text(diag)}"

    def get_messages(
        self, min_severity: DiagnosticSeverity = DiagnosticSeverity.WARNING
    ) -> list[str]:
        """Get formatted messages at or above the specified severity level."""
        messages = []
        for diag in self.diagnostics:
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
