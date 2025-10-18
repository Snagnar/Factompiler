# semantic.py
"""
Semantic analysis for the Factorio Circuit DSL.

This module performs:
- Symbol table construction and name resolution
- Type inference with implicit signal type allocation
- Mixed-type arithmetic validation and warnings
- Diagnostic collection and error reporting
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from dsl_compiler.src.dsl_ast import *
from dsl_compiler.src.signal_limits import MAX_IMPLICIT_VIRTUAL_SIGNALS

try:  # pragma: no cover - optional dependency
    from draftsman.data import signals as signal_data  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - fallback when draftsman data unavailable
    signal_data = None  # type: ignore[assignment]


EXPLAIN_MODE = False


def render_source_location(
    node: Optional[ASTNode], default_file: Optional[str] = None
) -> Optional[str]:
    """Format a human-friendly ``file:line`` string for an AST node."""

    if node is None:
        return None

    filename = getattr(node, "source_file", None) or default_file
    line = getattr(node, "line", 0) or 0

    if not filename and line <= 0:
        return None

    if filename and line > 0:
        return f"{Path(filename).name}:{line}"

    if filename:
        return Path(filename).name

    if line > 0:
        return f"?:{line}"

    return None


# =============================================================================
# Type System
# =============================================================================


@dataclass
class SignalTypeInfo:
    """Information about a signal type."""

    name: str  # e.g. "iron-plate", "signal-A", "__v1"
    is_implicit: bool = False  # True for compiler-allocated virtual signals
    is_virtual: bool = False  # True for Factorio virtual signals


@dataclass
class SignalValue:
    """A single-channel signal value."""

    signal_type: SignalTypeInfo
    count_expr: Optional[Expr] = None  # The expression that computes the count


@dataclass
class SignalDebugInfo:
    """Metadata describing a logical signal in the source program."""

    identifier: str
    signal_key: Optional[str]
    factorio_signal: Optional[str]
    source_node: ASTNode
    declared_type: Optional[str] = None
    location: Optional[str] = None
    category: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.identifier,
            "signal_key": self.signal_key,
            "factorio_signal": self.factorio_signal,
            "declared_type": self.declared_type,
            "location": self.location,
            "category": self.category,
            "source_ast": self.source_node,
        }


@dataclass
class IntValue:
    """A plain integer value."""

    value: Optional[int] = None  # None for computed values


@dataclass
class FunctionValue:
    """Value type for functions"""

    param_types: List["ValueInfo"] = field(default_factory=list)
    return_type: "ValueInfo" = field(default_factory=lambda: IntValue())


ValueInfo = Union[SignalValue, IntValue, FunctionValue]


@dataclass
class MemoryInfo:
    """Type information captured for memory declarations."""

    name: str
    symbol: "Symbol"
    signal_type: Optional[str] = None
    signal_info: Optional[SignalTypeInfo] = None
    explicit: bool = False


# =============================================================================
# Symbol Table
# =============================================================================


@dataclass
class Symbol:
    """Symbol table entry."""

    name: str
    symbol_type: (
        str  # "variable", "memory", "function", "parameter", "entity", "module"
    )
    value_type: ValueInfo
    defined_at: ASTNode
    is_mutable: bool = False
    properties: Optional[Dict[str, "Symbol"]] = None  # For modules and entities
    function_def: Optional[ASTNode] = None  # For functions - store AST for inlining
    debug_info: Dict[str, Any] = field(default_factory=dict)


class SymbolTable:
    """Hierarchical symbol table with scoping."""

    def __init__(self, parent: Optional["SymbolTable"] = None):
        self.parent = parent
        self.symbols: Dict[str, Symbol] = {}
        self.children: List["SymbolTable"] = []

    def define(self, symbol: Symbol) -> None:
        """Define a symbol in this scope."""
        if symbol.name in self.symbols:
            existing = self.symbols[symbol.name]
            raise SemanticError(
                f"Symbol '{symbol.name}' already defined", symbol.defined_at
            )
        self.symbols[symbol.name] = symbol

    def lookup(self, name: str) -> Optional[Symbol]:
        """Look up a symbol in this scope or parent scopes."""
        if name in self.symbols:
            return self.symbols[name]
        if self.parent:
            return self.parent.lookup(name)
        return None

    def create_child_scope(self) -> "SymbolTable":
        """Create a child scope."""
        child = SymbolTable(parent=self)
        self.children.append(child)
        return child


# =============================================================================
# Diagnostics
# =============================================================================


class DiagnosticLevel(Enum):
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

    def __post_init__(self):
        if self.node:
            self.line = self.node.line
            self.column = self.node.column


class DiagnosticCollector:
    """Collects and manages diagnostic messages."""

    def __init__(self):
        self.diagnostics: List[Diagnostic] = []
        self.error_count = 0
        self.warning_count = 0

    def info(self, message: str, node: Optional[ASTNode] = None) -> None:
        """Add an info diagnostic."""
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
        """Check if any errors were collected."""
        return self.error_count > 0

    def get_messages(self, level: Optional[DiagnosticLevel] = None) -> List[str]:
        """Get formatted diagnostic messages."""
        messages = []
        for diag in self.diagnostics:
            if level is None or diag.level == level:
                prefix = diag.level.value.upper()
                location = f"{diag.line}:{diag.column}" if diag.line > 0 else "?"
                messages.append(f"{prefix} [{location}]: {diag.message}")
        return messages


class SemanticError(Exception):
    """Exception raised for semantic analysis errors."""

    def __init__(self, message: str, node: Optional[ASTNode] = None):
        self.message = message
        self.node = node
        location = f" at {node.line}:{node.column}" if node and node.line > 0 else ""
        super().__init__(f"{message}{location}")


# =============================================================================
# Type Inference and Analysis
# =============================================================================


class SemanticAnalyzer(ASTVisitor):
    """Main semantic analysis visitor."""

    RESERVED_SIGNAL_RULES: Dict[str, Tuple[str, str]] = {
        "signal-W": ("error", "the memory write-enable channel"),
    }

    def __init__(self, strict_types: bool = False):
        self.strict_types = strict_types
        self.diagnostics = DiagnosticCollector()
        self.symbol_table = SymbolTable()
        self.current_scope = self.symbol_table

        # Type allocation
        self.implicit_type_counter = 0
        self.signal_type_map: Dict[str, str] = {}  # implicit -> factorio signal

        # Debug metadata
        self.signal_debug_info: Dict[str, SignalDebugInfo] = {}

        # Memory typing metadata
        self.memory_types: Dict[str, MemoryInfo] = {}

        # Expression type cache (using id() as key since AST nodes aren't hashable)
        self.expr_types: Dict[int, ValueInfo] = {}

        # Simple source tracking for optimizations (constants, entity outputs, etc.)
        self.simple_sources: set[str] = set()
        self.computed_values: set[str] = set()

    def mark_as_simple_source(self, name: str) -> None:
        """Record that a symbol originates from a simple source."""

        self.simple_sources.add(name)
        self.computed_values.discard(name)

    def mark_as_computed(self, name: str) -> None:
        """Record that a symbol results from computation."""

        self.computed_values.add(name)
        self.simple_sources.discard(name)

    def is_simple_source(self, name: str) -> bool:
        """Return True when a symbol still qualifies as a simple source."""

        return name in self.simple_sources and name not in self.computed_values

    def _track_source_nature(self, target: str, value_expr: Expr) -> None:
        """Update simple/computed tracking based on an assignment expression."""

        if isinstance(value_expr, (SignalLiteral, NumberLiteral)):
            self.mark_as_simple_source(target)
            return

        if isinstance(value_expr, PropertyAccessExpr):
            self.mark_as_simple_source(target)
            return

        if isinstance(value_expr, IdentifierExpr):
            if self.is_simple_source(value_expr.name):
                self.mark_as_simple_source(target)
            else:
                self.mark_as_computed(target)
            return

        if isinstance(value_expr, (BinaryOp, UnaryOp, CallExpr, ReadExpr, ProjectionExpr)):
            self.mark_as_computed(target)
            return

        # Fallback: treat as computed to be safe
        self.mark_as_computed(target)

    def _warn(self, message: str, node: Optional[ASTNode] = None) -> None:
        """Emit a warning, augmenting with explanations when requested."""

        if EXPLAIN_MODE and node is not None:
            explanation = self._get_warning_explanation(message, node)
            if explanation:
                message = f"{message}\n\nExplanation: {explanation}"

        self.diagnostics.warning(message, node)

    def _get_warning_explanation(
        self, message: str, node: Optional[ASTNode] = None
    ) -> Optional[str]:
        """Return an explanatory note for well-known warning patterns."""

        if "Mixed signal types" in message:
            return (
                "The compiler follows a left-operand-wins rule when arithmetic mixes"
                " signal channels. The resulting combinator keeps the left signal"
                " type because Factorio combinators can only output a single signal"
                " channel at a time. Use projections to align both operands or to"
                " choose the desired output channel explicitly."
            )

        return None

    def _emit_reserved_signal_diagnostic(
        self, signal_name: str, node: ASTNode, context: str
    ) -> None:
        rule = self.RESERVED_SIGNAL_RULES.get(signal_name)
        if not rule:
            return

        severity, reserved_context = rule
        message = (
            f"Signal '{signal_name}' is reserved for {reserved_context} and cannot be used {context}."
        )
        if severity == "error":
            self.diagnostics.error(message, node)
        else:
            self._warn(message, node)

    @staticmethod
    def _is_virtual_channel(signal_info: SignalTypeInfo) -> bool:
        if signal_info is None:
            return False
        if signal_info.is_virtual:
            return True
        name = signal_info.name
        return name.startswith("signal-") or name.startswith("__")

    def allocate_implicit_type(self) -> SignalTypeInfo:
        """Allocate a new implicit virtual signal type."""
        if self.implicit_type_counter >= MAX_IMPLICIT_VIRTUAL_SIGNALS:
            message = (
                "Ran out of compiler-allocated virtual signals "
                f"(limit {MAX_IMPLICIT_VIRTUAL_SIGNALS}). Reduce the program size."
            )
            self.diagnostics.error(message)
            raise SemanticError(message)

        self.implicit_type_counter += 1
        implicit_name = f"__v{self.implicit_type_counter}"
        factorio_signal = self._virtual_signal_name(self.implicit_type_counter)

        # Store mapping for debugging
        self.signal_type_map[implicit_name] = factorio_signal

        return SignalTypeInfo(name=implicit_name, is_implicit=True, is_virtual=True)

    def _virtual_signal_name(self, index: int) -> str:
        """Map implicit index to a unique Factorio virtual signal name."""
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        index -= 1  # Convert to zero-based
        name = ""
        while index >= 0:
            name = alphabet[index % 26] + name
            index = index // 26 - 1
        return f"signal-{name}"

    def _resolve_physical_signal_name(self, signal_key: Optional[str]) -> Optional[str]:
        if not signal_key:
            return None
        mapped = self.signal_type_map.get(signal_key)
        if isinstance(mapped, dict):
            return mapped.get("name", signal_key)
        if isinstance(mapped, str):
            return mapped
        return signal_key

    def _lookup_signal_category(self, signal_key: Optional[str]) -> Optional[str]:
        if not signal_key:
            return None
        mapped = self.signal_type_map.get(signal_key)
        if isinstance(mapped, dict):
            return mapped.get("type")
        return None

    def _register_signal_metadata(
        self,
        identifier: str,
        node: ASTNode,
        value_type: ValueInfo,
        declared_type: Optional[str] = None,
    ) -> Optional[SignalDebugInfo]:
        if not isinstance(value_type, SignalValue):
            return None

        signal_info = value_type.signal_type
        signal_key = getattr(signal_info, "name", None)
        factorio_signal = self._resolve_physical_signal_name(signal_key)
        location = render_source_location(node, getattr(node, "source_file", None))

        debug_info = SignalDebugInfo(
            identifier=identifier,
            signal_key=signal_key,
            factorio_signal=factorio_signal,
            source_node=node,
            declared_type=declared_type,
            location=location,
            category=self._lookup_signal_category(signal_key),
        )

        self.signal_debug_info[identifier] = debug_info

        symbol = self.current_scope.lookup(identifier)
        if symbol:
            symbol.debug_info["signal"] = debug_info

        return debug_info

    def get_signal_debug_payload(self, identifier: str) -> Optional[SignalDebugInfo]:
        return self.signal_debug_info.get(identifier)

    def make_signal_type_info(
        self, type_name: str, implicit: bool = False
    ) -> SignalTypeInfo:
        """Create a SignalTypeInfo with virtual flag inferred from name."""
        is_virtual = type_name.startswith("signal-") or type_name.startswith("__")
        return SignalTypeInfo(
            name=type_name, is_implicit=implicit, is_virtual=is_virtual
        )

    def is_valid_signal_type(self, signal_name: str) -> bool:
        """Validate that a signal identifier appears valid for Factorio use."""
        if not signal_name:
            return False

        if signal_name in self.signal_type_map:
            return True

        if signal_data is not None:
            if signal_name in signal_data.raw:
                return True
            if signal_name in signal_data.type_of:
                return True

        if signal_name.startswith("signal-"):
            return True

        # Permissive fallback for item/fluid names when no database available
        return True

    def get_expr_type(self, expr: Expr) -> ValueInfo:
        """Get the inferred type of an expression."""
        expr_id = id(expr)
        if expr_id in self.expr_types:
            return self.expr_types[expr_id]

        # Infer type
        value_type = self.infer_expr_type(expr)
        self.expr_types[expr_id] = value_type
        return value_type

    def infer_expr_type(self, expr: Expr) -> ValueInfo:
        """Infer the type of an expression."""
        if isinstance(expr, NumberLiteral):
            return IntValue(value=expr.value)

        elif isinstance(expr, StringLiteral):
            # Strings are only used as type literals, not values
            return IntValue()  # This shouldn't happen in normal cases

        elif isinstance(expr, DictLiteral):
            # Dictionary literals are primarily used for entity properties
            return IntValue()

        elif isinstance(expr, IdentifierExpr):
            symbol = self.current_scope.lookup(expr.name)
            if symbol is None:
                self.diagnostics.error(f"Undefined variable '{expr.name}'", expr)
                return IntValue()
            return symbol.value_type

        elif isinstance(expr, ReadExpr):
            # read(memory) returns the memory's signal type
            symbol = self.current_scope.lookup(expr.memory_name)
            if symbol is None:
                self.diagnostics.error(f"Undefined memory '{expr.memory_name}'", expr)
                return SignalValue(signal_type=self.allocate_implicit_type())
            if symbol.symbol_type != "memory":
                self.diagnostics.error(f"'{expr.memory_name}' is not a memory", expr)
            return symbol.value_type

        elif isinstance(expr, WriteExpr):
            if getattr(expr, "legacy_syntax", False):
                self.diagnostics.error(
                    "write(memory, value) has been replaced with write(value, memory, when=...).",
                    expr,
                )

            value_type = self.get_expr_type(expr.value)

            if getattr(expr, "when_once", False):
                enable_type = SignalValue(
                    signal_type=self.make_signal_type_info("signal-W", implicit=False)
                )
            elif expr.when is not None:
                enable_type = self.get_expr_type(expr.when)
            else:
                enable_type = SignalValue(
                    signal_type=self.make_signal_type_info("signal-W", implicit=False)
                )

            expr.enable_type = enable_type
            expr.value_type = value_type

            symbol = self.current_scope.lookup(expr.memory_name)
            if symbol is None:
                self.diagnostics.error(
                    f"Undefined memory '{expr.memory_name}' in write().", expr
                )
                return SignalValue(signal_type=self.allocate_implicit_type())

            if symbol.symbol_type != "memory":
                self.diagnostics.error(
                    f"'{expr.memory_name}' is not a memory symbol.", expr
                )
                return SignalValue(signal_type=self.allocate_implicit_type())

            if (
                expr.when is not None
                and not isinstance(enable_type, (SignalValue, IntValue))
            ):
                self.diagnostics.error(
                    "write when= argument must evaluate to a signal or integer.",
                    expr,
                )

            if not isinstance(value_type, SignalValue):
                self.diagnostics.error(
                    "write() expects a Signal value; provide a signal literal or projection.",
                    expr,
                )
                return symbol.value_type

            mem_info = self.memory_types.get(expr.memory_name)
            write_signal_name = (
                value_type.signal_type.name if value_type.signal_type else None
            )

            expected_type: Optional[str] = None
            if mem_info is not None and mem_info.signal_type:
                expected_type = mem_info.signal_type
            elif mem_info is None and isinstance(symbol.value_type, SignalValue):
                signal_info = symbol.value_type.signal_type
                if signal_info:
                    expected_type = signal_info.name

            # Infer storage type for implicitly declared memories the first time
            # we observe a concrete channel being written.
            if (
                expected_type is None
                and mem_info is not None
                and write_signal_name is not None
            ):
                inferred_info = self.make_signal_type_info(
                    write_signal_name,
                    implicit=write_signal_name.startswith("__"),
                )

                mem_info.signal_type = write_signal_name
                mem_info.signal_info = inferred_info

                if isinstance(symbol.value_type, SignalValue):
                    symbol.value_type.signal_type = inferred_info

                decl_node = getattr(mem_info.symbol, "defined_at", None)
                if isinstance(decl_node, MemDecl):
                    decl_node.signal_type = write_signal_name

                debug_info = self.signal_debug_info.get(expr.memory_name)
                if debug_info:
                    debug_info.signal_key = write_signal_name
                    debug_info.declared_type = write_signal_name
                    debug_info.factorio_signal = self._resolve_physical_signal_name(
                        write_signal_name
                    )
                    debug_info.category = self._lookup_signal_category(
                        write_signal_name
                    )

                expected_type = write_signal_name

            if expected_type is None:
                self.diagnostics.error(
                    f"Memory '{expr.memory_name}' does not have a resolved signal type.",
                    expr,
                )
                return symbol.value_type

            if write_signal_name is None:
                self.diagnostics.error(
                    f"Cannot determine signal type for value written to memory '{expr.memory_name}'.",
                    expr,
                )
                return symbol.value_type

            if write_signal_name != expected_type:
                message = (
                    f"Type mismatch: Memory '{expr.memory_name}' expects '{expected_type}'"
                    f" but write provides '{write_signal_name}'."
                )

                if mem_info is not None and not mem_info.explicit:
                    if self.strict_types:
                        self.diagnostics.error(message, expr)
                    else:
                        self._warn(message, expr)
                else:
                    self.diagnostics.error(message, expr)

            return symbol.value_type

        elif isinstance(expr, BinaryOp):
            return self.infer_binary_op_type(expr)

        elif isinstance(expr, UnaryOp):
            operand_type = self.get_expr_type(expr.expr)
            return operand_type  # Unary ops preserve type

        elif isinstance(expr, ProjectionExpr):
            # expr | "type" always returns Signal of specified type
            if expr.target_type in self.RESERVED_SIGNAL_RULES:
                self._emit_reserved_signal_diagnostic(
                    expr.target_type, expr, "as a projection target"
                )
            target_signal_type = SignalTypeInfo(name=expr.target_type)
            return SignalValue(signal_type=target_signal_type)

        elif isinstance(expr, SignalLiteral):
            # Signal literal: ("type", value) or just value
            if expr.signal_type:
                if expr.signal_type in self.RESERVED_SIGNAL_RULES:
                    self._emit_reserved_signal_diagnostic(
                        expr.signal_type, expr, "in signal literals"
                    )
                # Explicit type
                signal_type = SignalTypeInfo(name=expr.signal_type)
            else:
                # Implicit type
                signal_type = self.allocate_implicit_type()
            return SignalValue(signal_type=signal_type, count_expr=expr.value)

        elif isinstance(expr, CallExpr):
            self.visit_CallExpr(expr)

            builtin_type = self._infer_builtin_call_type(expr)
            if builtin_type is not None:
                return builtin_type

            func_symbol = self.current_scope.lookup(expr.name)
            if func_symbol and func_symbol.symbol_type == "function":
                return self._get_function_return_type(expr.name)

            return SignalValue(signal_type=self.allocate_implicit_type())

        elif isinstance(expr, PropertyAccess):
            # Entity or module property access
            object_symbol = self.current_scope.lookup(expr.object_name)
            if object_symbol is None:
                self.diagnostics.error(f"Undefined variable '{expr.object_name}'", expr)
                return IntValue()
            elif object_symbol.symbol_type == "entity":
                # Entity properties return signals for circuit control
                return SignalValue(signal_type=self.allocate_implicit_type())
            elif object_symbol.symbol_type == "module":
                # Module property access (functions)
                if (
                    object_symbol.properties
                    and expr.property_name in object_symbol.properties
                ):
                    func_symbol = object_symbol.properties[expr.property_name]
                    return func_symbol.value_type
                else:
                    self.diagnostics.error(
                        f"Module '{expr.object_name}' has no function '{expr.property_name}'",
                        expr,
                    )
                    return IntValue()
            else:
                self.diagnostics.error(
                    f"Cannot access property '{expr.property_name}' on '{expr.object_name}' of type {object_symbol.symbol_type}",
                    expr,
                )
                return IntValue()

        elif isinstance(expr, PropertyAccessExpr):
            # Entity or module property access in expression context
            object_symbol = self.current_scope.lookup(expr.object_name)
            if object_symbol is None:
                self.diagnostics.error(f"Undefined variable '{expr.object_name}'", expr)
                return IntValue()
            elif object_symbol.symbol_type == "entity":
                # Entity properties return signals for circuit control
                return SignalValue(signal_type=self.allocate_implicit_type())
            elif object_symbol.symbol_type == "module":
                # Module property access (functions)
                if (
                    object_symbol.properties
                    and expr.property_name in object_symbol.properties
                ):
                    func_symbol = object_symbol.properties[expr.property_name]
                    return func_symbol.value_type
                else:
                    self.diagnostics.error(
                        f"Module '{expr.object_name}' has no function '{expr.property_name}'",
                        expr,
                    )
                    return IntValue()
            else:
                self.diagnostics.error(
                    f"Cannot access property '{expr.property_name}' on '{expr.object_name}' of type {object_symbol.symbol_type}",
                    expr,
                )
                return IntValue()

        else:
            self.diagnostics.error(f"Unknown expression type: {type(expr)}", expr)
            return IntValue()

    def infer_binary_op_type(self, expr: BinaryOp) -> ValueInfo:
        """Infer type for binary operations with mixed-type rules."""
        left_type = self.get_expr_type(expr.left)
        right_type = self.get_expr_type(expr.right)

        # Comparison and logical operators yield virtual signals by default.
        # When a virtual operand participates, preserve that channel so explicit
        # projections (e.g. `| "signal-A"`) can become no-ops.
        if expr.op in ["==", "!=", "<", "<=", ">", ">=", "&&", "||"]:
            if isinstance(left_type, SignalValue) and self._is_virtual_channel(
                left_type.signal_type
            ):
                return left_type

            if isinstance(right_type, SignalValue) and self._is_virtual_channel(
                right_type.signal_type
            ):
                return right_type

            signal_type = self.allocate_implicit_type()
            return SignalValue(signal_type=signal_type)

        # Arithmetic operators: +, -, *, /, %
        if isinstance(left_type, IntValue) and isinstance(right_type, IntValue):
            # Int + Int = Int
            return IntValue()

        elif isinstance(left_type, SignalValue) and isinstance(right_type, IntValue):
            # Signal + Int = Signal (int coerced to signal's type)
            warning_msg = (
                f"Mixed types in binary operation at line {expr.line}:")
            warning_msg += (
                f"\n  Left operand:  '{left_type.signal_type.name}'"
                f"\n  Right operand: integer"
                f"\n  Result will keep signal '{left_type.signal_type.name}'"
                f"\n\n  Fix: Project the integer using ('{left_type.signal_type.name}', value)"
            )
            if self.strict_types:
                self.diagnostics.error(warning_msg, expr)
            else:
                self._warn(warning_msg, expr)
            return left_type

        elif isinstance(left_type, IntValue) and isinstance(right_type, SignalValue):
            # Int + Signal = Signal (int coerced to signal's type)
            warning_msg = (
                f"Mixed types in binary operation at line {expr.line}:")
            warning_msg += (
                f"\n  Left operand:  integer"
                f"\n  Right operand: '{right_type.signal_type.name}'"
                f"\n  Result will keep signal '{right_type.signal_type.name}'"
                f"\n\n  Fix: Wrap the integer as ('{right_type.signal_type.name}', value)"
            )
            if self.strict_types:
                self.diagnostics.error(warning_msg, expr)
            else:
                self._warn(warning_msg, expr)
            return right_type

        elif isinstance(left_type, SignalValue) and isinstance(right_type, SignalValue):
            # Signal + Signal
            if left_type.signal_type.name == right_type.signal_type.name:
                # Same type - can wire-merge or compute
                return left_type
            else:
                # Mixed types - left operand wins (with warning)
                warning_msg = (
                    f"Mixed signal types in binary operation at line {expr.line}:")
                warning_msg += (
                    f"\n  Left operand:  '{left_type.signal_type.name}' {expr.op}"
                    f"\n  Right operand: '{right_type.signal_type.name}'"
                    f"\n  Result will use left type: '{left_type.signal_type.name}'"
                    f"\n\n  To align types, consider:"
                        f"\n    - (left | \"{right_type.signal_type.name}\") {expr.op} right"
                        f"\n    - left {expr.op} (right | \"{left_type.signal_type.name}\")"
                        f"\n    - (... ) | \"desired-type\" to force an explicit channel"
                )

                if self.strict_types:
                    self.diagnostics.error(warning_msg, expr)
                else:
                    self._warn(warning_msg, expr)

                return left_type

        else:
            self.diagnostics.error(f"Invalid operand types for {expr.op}", expr)
            return IntValue()

    # =========================================================================
    # AST Visitor Methods
    # =========================================================================

    def visit(self, node: ASTNode) -> Any:
        """Visit a node and perform semantic analysis."""
        method_name = f"visit_{type(node).__name__}"
        if hasattr(self, method_name):
            return getattr(self, method_name)(node)
        else:
            return self.generic_visit(node)

    def visit_Program(self, node: Program) -> None:
        """Analyze the entire program."""
        for stmt in node.statements:
            self.visit(stmt)

    def visit_DeclStmt(self, node: DeclStmt) -> None:
        """Analyze typed declaration statement."""
        # Check that the value expression matches the declared type
        value_type = self.get_expr_type(node.value)

        # Special case: Signal x = 42; should create an implicit signal with quantity 42
        if node.type_name == "Signal" and isinstance(value_type, IntValue):
            # Create implicit signal with the integer value as quantity
            implicit_signal_type = self.allocate_implicit_type()
            value_type = SignalValue(
                signal_type=implicit_signal_type,
                count_expr=node.value,  # Store the integer expression as the count
            )

        # Convert type name to symbol type
        symbol_type = self._type_name_to_symbol_type(node.type_name)

        # Validate that the value matches the declared type
        if not self._value_matches_type(value_type, node.type_name):
            self.diagnostics.error(
                f"Cannot assign {self._value_type_name(value_type)} to {node.type_name} variable '{node.name}'",
                node,
            )

        # Define symbol with explicit type
        symbol = Symbol(
            name=node.name,
            symbol_type=symbol_type,
            value_type=value_type,
            defined_at=node,
        )
        try:
            self.current_scope.define(symbol)
        except SemanticError as e:
            self.diagnostics.error(e.message, node)
            return

        self._register_signal_metadata(node.name, node, value_type, node.type_name)
        self._track_source_nature(node.name, node.value)

    def _type_name_to_symbol_type(self, type_name: str) -> str:
        """Convert type name to symbol type."""
        if type_name == "Entity":
            return "entity"
        elif type_name == "Memory":
            return "memory"
        else:
            return "variable"

    def _value_matches_type(
        self, value_type: ValueInfo, expected_type_name: str
    ) -> bool:
        """Check if a value type matches the expected type name."""
        type_map = {
            "int": IntValue,
            "Signal": SignalValue,
            "SignalType": SignalValue,  # For now, treat as Signal
            "Entity": SignalValue,  # Entity calls return signals for now
            "Memory": SignalValue,  # Memory is stored as Signal type
        }
        return (
            isinstance(value_type, type_map.get(expected_type_name, type(None)))
            or expected_type_name not in type_map
        )

    def _value_type_name(self, value_type: ValueInfo) -> str:
        """Get a human-readable name for a value type."""
        type_names = {
            IntValue: "int",
            SignalValue: "Signal",
            FunctionValue: "function",
        }
        return type_names.get(type(value_type), "unknown")

    def _function_returns_entity(self, function_name: str) -> bool:
        """Check if a function returns an entity by examining its definition."""
        from dsl_compiler.src.dsl_ast import ReturnStmt, CallExpr

        func_symbol = self.current_scope.lookup(function_name)
        if not func_symbol or func_symbol.symbol_type != "function":
            return False

        # If the function definition has return statements with place() calls
        if hasattr(func_symbol, "function_def") and func_symbol.function_def:
            for stmt in func_symbol.function_def.body:
                if isinstance(stmt, ReturnStmt) and stmt.expr:
                    if isinstance(stmt.expr, CallExpr) and stmt.expr.name == "place":
                        return True

        return False

    def _get_function_return_type(self, function_name: str) -> ValueInfo:
        """Determine the return type of a function by analyzing its return statements."""
        from dsl_compiler.src.dsl_ast import ReturnStmt

        func_symbol = self.current_scope.lookup(function_name)
        if not func_symbol or func_symbol.symbol_type != "function":
            return SignalValue(signal_type=self.allocate_implicit_type())

        # If the function definition has return statements, analyze them
        if hasattr(func_symbol, "function_def") and func_symbol.function_def:
            for stmt in func_symbol.function_def.body:
                if isinstance(stmt, ReturnStmt) and stmt.expr:
                    # Get the type of the return expression
                    return_type = self.get_expr_type(stmt.expr)
                    return return_type

        # Default to signal type if no return statements found
        return SignalValue(signal_type=self.allocate_implicit_type())

    def _infer_parameter_type(self, param_name: str, func_def) -> ValueInfo:
        """Infer parameter type from usage within the function body."""
        # With bundles removed, parameters default to signal types unless explicitly annotated
        return SignalValue(signal_type=self.allocate_implicit_type())

    def _expression_uses_identifier(self, expr, identifier_name: str) -> bool:
        """Check if an expression uses a specific identifier."""
        from dsl_compiler.src.dsl_ast import (
            IdentifierExpr,
            BinaryOp,
            CallExpr,
            PropertyAccess,
        )

        if isinstance(expr, IdentifierExpr):
            return expr.name == identifier_name
        elif isinstance(expr, BinaryOp):
            return self._expression_uses_identifier(
                expr.left, identifier_name
            ) or self._expression_uses_identifier(expr.right, identifier_name)
        elif isinstance(expr, CallExpr):
            return any(
                self._expression_uses_identifier(arg, identifier_name)
                for arg in expr.args
            )
        elif isinstance(expr, PropertyAccess):
            return expr.object_name == identifier_name
        # Add more expression types as needed
        return False

    def _expression_involves_parameter(self, expr) -> bool:
        """Check if an expression involves any function parameters."""
        from dsl_compiler.src.dsl_ast import (
            IdentifierExpr,
            BinaryOp,
            CallExpr,
            PropertyAccess,
        )

        if isinstance(expr, IdentifierExpr):
            # Check if this identifier is a parameter
            symbol = self.current_scope.lookup(expr.name)
            return symbol and symbol.symbol_type == "parameter"
        elif isinstance(expr, BinaryOp):
            return self._expression_involves_parameter(
                expr.left
            ) or self._expression_involves_parameter(expr.right)
        elif isinstance(expr, CallExpr):
            return any(self._expression_involves_parameter(arg) for arg in expr.args)
        elif isinstance(expr, PropertyAccess):
            return self._expression_involves_parameter_name(expr.object_name)
        return False

    def _expression_involves_parameter_name(self, name: str) -> bool:
        """Check if a name refers to a parameter."""
        symbol = self.current_scope.lookup(name)
        return symbol and symbol.symbol_type == "parameter"

    def _binary_op_involves_parameters(self, expr) -> bool:
        """Check if a binary operation involves function parameters."""
        from dsl_compiler.src.dsl_ast import BinaryOp

        if isinstance(expr, BinaryOp):
            return self._expression_involves_parameter(
                expr.left
            ) or self._expression_involves_parameter(expr.right)
        return False

    def _infer_builtin_call_type(self, expr: CallExpr) -> Optional[ValueInfo]:
        """Return ValueInfo for built-in calls or None if not handled."""
        if expr.name == "place":
            signal_type = self.allocate_implicit_type()
            expr.metadata["entity_signal_type"] = signal_type
            return SignalValue(signal_type=signal_type)

        if expr.name == "memory":
            signal_type = self._resolve_memory_signal_type(expr)
            expr.metadata["resolved_signal_type"] = signal_type
            return SignalValue(signal_type=signal_type)

        return None

    def _resolve_memory_signal_type(self, expr: CallExpr) -> SignalTypeInfo:
        """Determine the signal type for memory() calls."""
        explicit_type = None
        if len(expr.args) >= 2 and isinstance(expr.args[1], StringLiteral):
            explicit_type = expr.args[1].value

        if explicit_type:
            return self.make_signal_type_info(explicit_type, implicit=False)

        if expr.args:
            initial_type = self.get_expr_type(expr.args[0])
            if isinstance(initial_type, SignalValue):
                return initial_type.signal_type

        return self.allocate_implicit_type()

    def _validate_builtin_call(self, node: CallExpr) -> None:
        """Semantic validation for built-in calls like place/input/memory."""
        if node.name == "place":
            if not (3 <= len(node.args) <= 4):
                self.diagnostics.error(
                    "place() requires 3 or 4 arguments (prototype, x, y, [properties])",
                    node,
                )
                return

            prototype = node.args[0]
            if not isinstance(prototype, StringLiteral):
                self.diagnostics.error(
                    "place() prototype must be a string literal",
                    prototype if node.args else node,
                )

            # Analyze coordinate expressions
            if len(node.args) >= 2:
                self.get_expr_type(node.args[1])
            if len(node.args) >= 3:
                self.get_expr_type(node.args[2])

            if len(node.args) == 4:
                if not isinstance(node.args[3], DictLiteral):
                    self.diagnostics.error(
                        "place() properties must be a dictionary literal",
                        node.args[3],
                    )
                else:
                    # Analyze property values
                    for value in node.args[3].entries.values():
                        self.get_expr_type(value)

            if isinstance(prototype, StringLiteral):
                node.metadata["prototype"] = prototype.value

        elif node.name == "input":
            self.diagnostics.error(
                "input() helper has been removed; use explicit signal literals or direct wiring instead.",
                node,
            )
            for arg in node.args:
                self.get_expr_type(arg)

        elif node.name == "memory":
            if not (1 <= len(node.args) <= 2):
                self.diagnostics.error(
                    "memory() expects one or two arguments (initial, [type])",
                    node,
                )

            if len(node.args) >= 1:
                self.get_expr_type(node.args[0])

            if len(node.args) == 2 and not isinstance(node.args[1], StringLiteral):
                self.diagnostics.error(
                    "memory() explicit type must be a string literal",
                    node.args[1],
                )
            elif len(node.args) == 2:
                node.metadata["explicit_type"] = node.args[1].value

        else:
            # Generic fallback - analyze all arguments
            for arg in node.args:
                self.get_expr_type(arg)

    def visit_MemDecl(self, node: MemDecl) -> None:
        """Analyze memory declaration."""
        declared_type = node.signal_type

        if declared_type is None:
            # Allocate a compiler-managed virtual channel placeholder. The
            # concrete storage channel will be inferred from the first write().
            signal_info = self.allocate_implicit_type()
        else:
            if not self.is_valid_signal_type(declared_type):
                self.diagnostics.error(
                    f"Invalid signal type '{declared_type}' for memory '{node.name}'",
                    node,
                )
            elif declared_type in self.RESERVED_SIGNAL_RULES:
                self._emit_reserved_signal_diagnostic(
                    declared_type, node, "for memory storage"
                )

            signal_info = self.make_signal_type_info(declared_type)

        memory_type = SignalValue(signal_type=signal_info)

        symbol = Symbol(
            name=node.name,
            symbol_type="memory",
            value_type=memory_type,
            defined_at=node,
            is_mutable=True,
        )
        try:
            self.current_scope.define(symbol)
        except SemanticError as e:
            self.diagnostics.error(e.message, node)
            return

        mem_info = MemoryInfo(
            name=node.name,
            symbol=symbol,
            signal_type=declared_type,
            signal_info=signal_info,
            explicit=declared_type is not None,
        )
        self.memory_types[node.name] = mem_info

        # Handle inline initialization hints
        if node.init_expr is not None:
            init_type = self.get_expr_type(node.init_expr)

            if isinstance(init_type, SignalValue) and init_type.signal_type is not None:
                init_signal_name = init_type.signal_type.name

                if mem_info.signal_type is None:
                    mem_info.signal_type = init_signal_name
                    mem_info.signal_info = init_type.signal_type
                    memory_type.signal_type = init_type.signal_type
                    symbol.value_type.signal_type = init_type.signal_type
                    node.signal_type = init_signal_name
                    declared_type = init_signal_name
                elif init_signal_name != mem_info.signal_type:
                    self._warn(
                        (
                            f"Memory '{node.name}' initialization writes '{init_signal_name}'"
                            f" but declaration expects '{mem_info.signal_type}'."
                            " Project the initializer to the declared type or update the declaration."
                        ),
                        node.init_expr,
                    )
            elif isinstance(init_type, IntValue) and mem_info.signal_type is None:
                # Preserve implicit virtual channel for integer initializers until a
                # concrete signal type is inferred by a later write.
                pass

        self._register_signal_metadata(
            node.name,
            node,
            memory_type,
            declared_type,
        )

    def visit_FuncDecl(self, node: FuncDecl) -> None:
        """Analyze function declaration."""
        # Analyze function parameters and infer their types
        param_types = []
        for param_name in node.params:
            # For now, assume all parameters are IntValue
            # A more sophisticated implementation would infer from usage
            param_types.append(IntValue())

        # Analyze return type by examining return statements in the body
        return_type = self._infer_function_return_type(node.body)

        # Create function symbol with proper function type
        func_symbol = Symbol(
            name=node.name,
            symbol_type="function",
            value_type=FunctionValue(param_types=param_types, return_type=return_type),
            defined_at=node,
            function_def=node,  # Store AST for analysis
        )
        try:
            self.current_scope.define(func_symbol)
        except SemanticError as e:
            self.diagnostics.error(e.message, node)
            return  # Skip processing function body if definition failed

        # Create new scope for function body
        func_scope = self.current_scope.create_child_scope()
        old_scope = self.current_scope
        self.current_scope = func_scope

        # Define parameters with inferred types
        for param_name in node.params:
            # Try to infer parameter type from usage within the function
            param_type = self._infer_parameter_type(param_name, node)
            param_symbol = Symbol(
                name=param_name,
                symbol_type="parameter",
                value_type=param_type,
                defined_at=node,
            )
            try:
                self.current_scope.define(param_symbol)
            except SemanticError as e:
                self.diagnostics.error(e.message, node)

        # Analyze function body
        for stmt in node.body:
            self.visit(stmt)

        # Restore scope
        self.current_scope = old_scope

    def _infer_function_return_type(self, body: List[Statement]) -> ValueInfo:
        """Infer function return type by analyzing return statements."""
        from dsl_compiler.src.dsl_ast import ReturnStmt

        # Look for return statements
        for stmt in body:
            if isinstance(stmt, ReturnStmt) and stmt.expr:
                # Try to infer the type of the return expression
                # For now, return IntValue as a safe default
                # A more sophisticated implementation would analyze the expression
                return IntValue()

        # No return statement found, assume void (represented as IntValue for now)
        return IntValue()

    def visit_ExprStmt(self, node: ExprStmt) -> None:
        """Analyze expression statement."""
        self.get_expr_type(node.expr)

    def visit_AssignStmt(self, node: AssignStmt) -> None:
        """Analyze assignment statement."""
        # Check target exists and is mutable
        target_symbol = None
        if isinstance(node.target, Identifier):
            symbol = self.current_scope.lookup(node.target.name)
            if symbol is None:
                # Check if this is entity assignment (place() call)
                if isinstance(node.value, CallExpr) and node.value.name == "place":
                    # Create entity symbol
                    entity_symbol = Symbol(
                        name=node.target.name,
                        symbol_type="entity",
                        value_type=IntValue(),  # Entities don't have specific types
                        defined_at=node.target,
                        is_mutable=True,
                    )
                    try:
                        self.current_scope.define(entity_symbol)
                    except SemanticError as e:
                        self.diagnostics.error(e.message, node.target)
                else:
                    self.diagnostics.error(
                        f"Undefined variable '{node.target.name}'", node.target
                    )
            elif not symbol.is_mutable and symbol.symbol_type != "entity":
                self.diagnostics.error(
                    f"Cannot assign to immutable '{node.target.name}'", node.target
                )
            else:
                target_symbol = symbol
        elif isinstance(node.target, PropertyAccess):
            # Check that the object exists
            object_symbol = self.current_scope.lookup(node.target.object_name)
            if object_symbol is None:
                self.diagnostics.error(
                    f"Undefined entity '{node.target.object_name}'", node.target
                )
            elif object_symbol.symbol_type != "entity":
                self.diagnostics.error(
                    f"Cannot access property '{node.target.property_name}' on non-entity '{node.target.object_name}'",
                    node.target,
                )

        # Type check assignment
        value_type = self.get_expr_type(node.value)
        # Additional type compatibility checks could go here

        if isinstance(node.target, Identifier):
            self._register_signal_metadata(
                node.target.name,
                node.target,
                value_type,
                target_symbol.symbol_type if target_symbol else None,
            )
            self._track_source_nature(node.target.name, node.value)

    def visit_ImportStmt(self, node: ImportStmt) -> None:
        """Analyze import statement."""
        # With C-style preprocessing, import statements should have been
        # replaced with the actual imported content, so this should rarely be called.
        # If we do encounter an import statement, it means the file wasn't found
        # during preprocessing, so we'll just log a warning.
        self.diagnostics.error(
            f"Import statement found in AST - file may not have been found during preprocessing: {node.path}",
            node,
        )

    def visit_ReturnStmt(self, node: ReturnStmt) -> None:
        """Analyze return statement."""
        # Type check the return expression
        if node.expr:
            return_type = self.get_expr_type(node.expr)
            # Could check against function return type here

    def visit_CallExpr(self, node: CallExpr) -> None:
        """Analyze function call expressions."""
        builtin_names = {"place", "input", "memory"}

        func_symbol = self.current_scope.lookup(node.name)
        if func_symbol is None:
            if node.name in builtin_names:
                self._validate_builtin_call(node)
                return
            self.diagnostics.error(f"Undefined function '{node.name}'", node)
            return

        if func_symbol.symbol_type != "function":
            self.diagnostics.error(f"'{node.name}' is not a function", node)
            return

        # User-defined function call
        if hasattr(func_symbol, "function_def") and func_symbol.function_def:
            expected_params = len(func_symbol.function_def.params)
            actual_args = len(node.args)
            if actual_args != expected_params:
                self.diagnostics.error(
                    f"Function '{node.name}' expects {expected_params} arguments, got {actual_args}",
                    node,
                )

        for arg in node.args:
            self.get_expr_type(arg)

    def visit_DictLiteral(self, node: DictLiteral) -> None:
        """Analyze dictionary literal entries."""
        for value in node.entries.values():
            self.get_expr_type(value)

    def generic_visit(self, node: ASTNode) -> Any:
        """Default visitor - traverse children."""
        for field_name, field_value in node.__dict__.items():
            if isinstance(field_value, ASTNode):
                self.visit(field_value)
            elif isinstance(field_value, list):
                for item in field_value:
                    if isinstance(item, ASTNode):
                        self.visit(item)


# =============================================================================
# Public API
# =============================================================================


def analyze_program(
    program: Program,
    strict_types: bool = False,
    analyzer: Optional["SemanticAnalyzer"] = None,
    file_path: Optional[str] = None,
) -> DiagnosticCollector:
    """Perform semantic analysis on a program AST."""
    if analyzer is None:
        analyzer = SemanticAnalyzer(strict_types=strict_types)

    # Set current file directory for import resolution
    if file_path:
        analyzer.current_file_dir = Path(file_path).parent
    else:
        analyzer.current_file_dir = Path("tests/sample_programs")  # Default for tests

    analyzer.visit(program)
    return analyzer.diagnostics


def analyze_file(file_path: str, strict_types: bool = False) -> DiagnosticCollector:
    """Analyze a DSL file."""
    from dsl_compiler.src.parser import DSLParser

    parser = DSLParser()
    try:
        program = parser.parse_file(Path(file_path))
        return analyze_program(program, strict_types)
    except Exception as e:
        diagnostics = DiagnosticCollector()
        diagnostics.error(f"Failed to parse {file_path}: {e}")
        return diagnostics
