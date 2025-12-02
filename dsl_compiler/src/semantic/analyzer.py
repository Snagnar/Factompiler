"""Semantic analysis for the Factorio Circuit DSL."""

from typing import Any, Dict, List, Optional, Tuple

from dsl_compiler.src.ast.expressions import (
    IdentifierExpr,
    ProjectionExpr,
    PropertyAccessExpr,
    ReadExpr,
    SignalLiteral,
    UnaryOp,
    WriteExpr,
    ASTNode,
    Expr,
    CallExpr,
    BinaryOp,
)
from dsl_compiler.src.ast.literals import (
    DictLiteral,
    NumberLiteral,
    PropertyAccess,
    StringLiteral,
    Identifier,
)
from dsl_compiler.src.ast.statements import (
    ReturnStmt,
    MemDecl,
    Program,
    DeclStmt,
    FuncDecl,
    Statement,
    AssignStmt,
    ExprStmt,
    ImportStmt,
)
from dsl_compiler.src.ast.base import ASTVisitor
from dsl_compiler.src.common.diagnostics import (
    ProgramDiagnostics,
)
from dsl_compiler.src.common.signal_registry import (
    SignalTypeRegistry,
    is_valid_factorio_signal,
)
from dsl_compiler.src.common.source_location import (
    SourceLocation,
)
from dsl_compiler.src.common.symbol_types import SymbolType

from .exceptions import SemanticError
from .signal_allocator import SignalAllocator
from .symbol_table import SymbolTable, Symbol
from .type_system import (
    FunctionValue,
    IntValue,
    MemoryInfo,
    SignalDebugInfo,
    SignalTypeInfo,
    SignalValue,
    ValueInfo,
)


class SemanticAnalyzer(ASTVisitor):
    """Main semantic analysis visitor."""

    RESERVED_SIGNAL_RULES: Dict[str, Tuple[str, str]] = {
        "signal-W": ("error", "the memory write-enable channel"),
    }

    def __init__(self, diagnostics: ProgramDiagnostics, strict_types: bool = False):
        self.strict_types = strict_types
        self.diagnostics = diagnostics
        self.diagnostics.default_stage = "semantic"
        self.symbol_table = SymbolTable()
        self.current_scope = self.symbol_table

        # Type allocation - use centralized registry
        self.signal_registry = SignalTypeRegistry()
        self.signal_allocator = SignalAllocator(self.signal_registry)

        # Debug metadata
        self.signal_debug_info: Dict[str, SignalDebugInfo] = {}

        # Memory typing metadata
        self.memory_types: Dict[str, MemoryInfo] = {}

        # Expression type cache (using id() as key since AST nodes aren't hashable)
        self.expr_types: Dict[int, ValueInfo] = {}

        # Simple source tracking for optimizations (constants, entity outputs, etc.)
        self.simple_sources: set[str] = set()
        self.computed_values: set[str] = set()

    @property
    def signal_type_map(self) -> Dict[str, Any]:
        """Get signal type map from registry."""
        return self.signal_registry.get_all_mappings()

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

        if isinstance(
            value_expr, (BinaryOp, UnaryOp, CallExpr, ReadExpr, ProjectionExpr)
        ):
            self.mark_as_computed(target)
            return

        # Fallback: treat as computed to be safe
        self.mark_as_computed(target)

    def _emit_reserved_signal_diagnostic(
        self, signal_name: str, node: ASTNode, context: str
    ) -> None:
        rule = self.RESERVED_SIGNAL_RULES.get(signal_name)
        if not rule:
            return

        severity, reserved_context = rule
        message = f"Signal '{signal_name}' is reserved for {reserved_context} and cannot be used {context}."
        if severity == "error":
            self.diagnostics.error(message, stage="semantic", node=node)
        else:
            self.diagnostics.warning(message, stage="semantic", node=node)

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
        return self.signal_allocator.allocate_implicit_type()

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
        location = SourceLocation.render(node, getattr(node, "source_file", None))

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

    def validate_signal_type_with_error(
        self, signal_name: str, node: ASTNode, context: str = ""
    ) -> bool:
        """Validate a signal type and emit an error if invalid.

        Args:
            signal_name: The signal name to validate
            node: The AST node for error reporting
            context: Optional context string for the error message

        Returns:
            True if the signal is valid, False otherwise
        """
        if not signal_name:
            return False

        # Check if already registered in our type map
        if signal_name in self.signal_type_map:
            return True

        is_valid, error_msg = is_valid_factorio_signal(signal_name)
        if not is_valid:
            context_str = f" {context}" if context else ""
            self.diagnostics.error(
                f"{error_msg}{context_str}",
                stage="semantic",
                node=node,
            )
        return is_valid

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
                self.diagnostics.error(
                    f"Undefined variable '{expr.name}'", stage="semantic", node=expr
                )
                return IntValue()
            return symbol.value_type

        elif isinstance(expr, ReadExpr):
            # read(memory) returns the memory's signal type
            symbol = self.current_scope.lookup(expr.memory_name)
            if symbol is None:
                self.diagnostics.error(
                    f"Undefined memory '{expr.memory_name}'",
                    stage="semantic",
                    node=expr,
                )
                return SignalValue(signal_type=self.allocate_implicit_type())
            if symbol.symbol_type != SymbolType.MEMORY:
                self.diagnostics.error(
                    f"'{expr.memory_name}' is not a memory", stage="semantic", node=expr
                )
            return symbol.value_type

        elif isinstance(expr, WriteExpr):
            value_type = self.get_expr_type(expr.value)

            if expr.when is not None:
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
                    f"Undefined memory '{expr.memory_name}' in write().",
                    stage="semantic",
                    node=expr,
                )
                return SignalValue(signal_type=self.allocate_implicit_type())

            if symbol.symbol_type != SymbolType.MEMORY:
                self.diagnostics.error(
                    f"'{expr.memory_name}' is not a memory symbol.",
                    stage="semantic",
                    node=expr,
                )
                return SignalValue(signal_type=self.allocate_implicit_type())

            if expr.when is not None and not isinstance(
                enable_type, (SignalValue, IntValue)
            ):
                self.diagnostics.error(
                    "write when= argument must evaluate to a signal or integer.",
                    stage="semantic",
                    node=expr,
                )

            # Convert IntValue to SignalValue for write() value argument
            if isinstance(value_type, IntValue):
                # Allocate implicit signal type for bare integer constants in write()
                implicit_signal_type = self.allocate_implicit_type()
                value_type = SignalValue(
                    signal_type=implicit_signal_type,
                    count_expr=expr.value,
                )
                expr.value_type = value_type

            if not isinstance(value_type, SignalValue):
                self.diagnostics.error(
                    "write() expects a Signal value; provide a signal literal or projection.",
                    stage="semantic",
                    node=expr,
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
                    stage="semantic",
                    node=expr,
                )
                return symbol.value_type

            if write_signal_name is None:
                self.diagnostics.error(
                    f"Cannot determine signal type for value written to memory '{expr.memory_name}'.",
                    stage="semantic",
                    node=expr,
                )
                return symbol.value_type

            if write_signal_name != expected_type:
                message = (
                    f"Type mismatch: Memory '{expr.memory_name}' expects '{expected_type}'"
                    f" but write provides '{write_signal_name}'."
                )

                # Implicit types (__v*) from function parameters are polymorphic -
                # they take on the actual type passed at the call site, so we don't
                # error when writing them to explicitly-typed memories
                is_implicit_type = write_signal_name.startswith("__v")

                if is_implicit_type:
                    # Implicit types are polymorphic, allow without warning
                    pass
                elif mem_info is not None and not mem_info.explicit:
                    if self.strict_types:
                        self.diagnostics.error(message, stage="semantic", node=expr)
                    else:
                        self.diagnostics.warning(message, stage="semantic", node=expr)
                else:
                    self.diagnostics.error(message, stage="semantic", node=expr)

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
            # Validate that the target type is a valid Factorio signal
            self.validate_signal_type_with_error(
                expr.target_type, expr, "in projection"
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
                # Validate that the signal type is a valid Factorio signal
                self.validate_signal_type_with_error(
                    expr.signal_type, expr, "in signal literal"
                )
                # Explicit type
                signal_type = SignalTypeInfo(name=expr.signal_type)
                return SignalValue(signal_type=signal_type, count_expr=expr.value)
            else:
                # Bare number literal (signal_type is None)
                # These are syntactic sugar: "7" is parsed as SignalLiteral(NumberLiteral(7), None)
                # Treat as integer constant for type propagation (Int + Signal -> Signal)
                # Only allocate implicit signal type when used in Signal declaration context
                if isinstance(expr.value, NumberLiteral):
                    return IntValue()
                else:
                    # Non-number expression without explicit type - allocate implicit type
                    signal_type = self.allocate_implicit_type()
                    return SignalValue(signal_type=signal_type, count_expr=expr.value)

        elif isinstance(expr, CallExpr):
            self.visit_CallExpr(expr)

            builtin_type = self._infer_builtin_call_type(expr)
            if builtin_type is not None:
                return builtin_type

            func_symbol = self.current_scope.lookup(expr.name)
            if func_symbol and func_symbol.symbol_type == SymbolType.FUNCTION:
                return self._get_function_return_type(expr.name)

            return SignalValue(signal_type=self.allocate_implicit_type())

        elif isinstance(expr, PropertyAccess):
            # Entity or module property access
            object_symbol = self.current_scope.lookup(expr.object_name)
            if object_symbol is None:
                self.diagnostics.error(
                    f"Undefined variable '{expr.object_name}'",
                    stage="semantic",
                    node=expr,
                )
                return IntValue()
            elif object_symbol.symbol_type == SymbolType.ENTITY:
                # Entity properties return signals for circuit control
                return SignalValue(signal_type=self.allocate_implicit_type())
            elif object_symbol.symbol_type == SymbolType.MODULE:
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
                        stage="semantic",
                        node=expr,
                    )
                    return IntValue()
            else:
                self.diagnostics.error(
                    f"Cannot access property '{expr.property_name}' on '{expr.object_name}' of type {object_symbol.symbol_type}",
                    stage="semantic",
                    node=expr,
                )
                return IntValue()

        elif isinstance(expr, PropertyAccessExpr):
            # Entity or module property access in expression context
            object_symbol = self.current_scope.lookup(expr.object_name)
            if object_symbol is None:
                self.diagnostics.error(
                    f"Undefined variable '{expr.object_name}'",
                    stage="semantic",
                    node=expr,
                )
                return IntValue()
            elif object_symbol.symbol_type == SymbolType.ENTITY:
                # Entity properties return signals for circuit control
                return SignalValue(signal_type=self.allocate_implicit_type())
            elif object_symbol.symbol_type == SymbolType.MODULE:
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
                        stage="semantic",
                        node=expr,
                    )
                    return IntValue()
            else:
                self.diagnostics.error(
                    f"Cannot access property '{expr.property_name}' on '{expr.object_name}' of type {object_symbol.symbol_type}",
                    stage="semantic",
                    node=expr,
                )
                return IntValue()

        else:
            self.diagnostics.error(
                f"Unknown expression type: {type(expr)}", stage="semantic", node=expr
            )
            return IntValue()

    def _check_signal_type_compatibility(
        self,
        left_type: ValueInfo,
        right_type: ValueInfo,
        op: str,
        node: ASTNode,
    ) -> tuple[ValueInfo, Any]:
        """Check type compatibility for arithmetic operations and return result type with optional warning.

        Returns: (result_type, warning_message)
        """
        # Int + Int = Int
        if isinstance(left_type, IntValue) and isinstance(right_type, IntValue):
            return IntValue(), None

        # Signal + Int = Signal (int coerced to signal's type)
        # No warning - this is a common and expected operation
        if isinstance(left_type, SignalValue) and isinstance(right_type, IntValue):
            return left_type, None

        # Int + Signal = Signal (int coerced to signal's type)
        # No warning - this is a common and expected operation
        if isinstance(left_type, IntValue) and isinstance(right_type, SignalValue):
            return right_type, None

        # Signal + Signal
        if isinstance(left_type, SignalValue) and isinstance(right_type, SignalValue):
            # Same type - can wire-merge or compute
            if left_type.signal_type.name == right_type.signal_type.name:
                return left_type, None

            # Both virtual signals - no warning (virtual signals are interchangeable)
            left_is_virtual = self._is_virtual_channel(left_type.signal_type)
            right_is_virtual = self._is_virtual_channel(right_type.signal_type)
            if left_is_virtual and right_is_virtual:
                return left_type, None

            # Mixed types - left operand wins (with warning)
            warning_msg = f"Mixed signal types in binary operation at line {node.line}:"
            warning_msg += (
                f"\n  Left operand:  '{left_type.signal_type.name}' {op}"
                f"\n  Right operand: '{right_type.signal_type.name}'"
                f"\n  Result will use left type: '{left_type.signal_type.name}'"
                f"\n\n  To align types, consider:"
                f'\n    - (left | "{right_type.signal_type.name}") {op} right'
                f'\n    - left {op} (right | "{left_type.signal_type.name}")'
                f'\n    - (... ) | "desired-type" to force an explicit channel'
            )
            return left_type, warning_msg

        # Invalid operand types
        return IntValue(), f"Invalid operand types for {op}"

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

        # Arithmetic operators: use extracted compatibility checker
        result_type, warning_msg = self._check_signal_type_compatibility(
            left_type, right_type, expr.op, expr
        )

        if warning_msg:
            if self.strict_types:
                self.diagnostics.error(warning_msg, stage="semantic", node=expr)
            else:
                self.diagnostics.warning(warning_msg, stage="semantic", node=expr)

        return result_type

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

    def _try_simplify_signal_projection(
        self, proj_expr: ProjectionExpr
    ) -> Optional[SignalLiteral]:
        """
        Try to simplify a projection of a literal to a direct signal literal.

        Transforms:
        - 50 | "signal-A" → ("signal-A", 50)
        - ("copper-plate", 10) | "iron-plate" → ("iron-plate", 10)
        - ((50 | "A") | "B") | "C" → ("C", 50) (recursive simplification)

        This ensures we materialize only one constant combinator instead of
        constant + arithmetic combinator.

        Returns:
            SignalLiteral if simplification is safe, None otherwise.
        """
        source = proj_expr.expr
        target_type = proj_expr.target_type

        # Case 1: NumberLiteral projection (e.g., 50 | "signal-A")
        if isinstance(source, NumberLiteral):
            return SignalLiteral(
                value=source,
                signal_type=target_type,
                line=proj_expr.line,
                column=proj_expr.column,
                raw_text=proj_expr.raw_text,
            )

        # Case 2: SignalLiteral projection (e.g., ("copper", 50) | "iron-plate")
        # Extract the inner value and apply the new type
        if isinstance(source, SignalLiteral):
            return SignalLiteral(
                value=source.value,  # Keep the original value expression
                signal_type=target_type,
                line=proj_expr.line,
                column=proj_expr.column,
                raw_text=proj_expr.raw_text,
            )

        # Case 3: Nested projection (e.g., (50 | "A") | "B")
        # Recursively simplify the inner projection first
        if isinstance(source, ProjectionExpr):
            inner_simplified = self._try_simplify_signal_projection(source)
            if inner_simplified is not None:
                # Inner simplified successfully, now apply outer projection
                return SignalLiteral(
                    value=inner_simplified.value,
                    signal_type=target_type,
                    line=proj_expr.line,
                    column=proj_expr.column,
                    raw_text=proj_expr.raw_text,
                )

        # Case 4: Identifier projection - NOT simplified
        # Even with single consumer, we don't simplify to preserve user expectations.
        # If user writes: Signal a = 5; Signal b = a | "type";
        # They expect 'a' to be materialized, not optimized away.

        # Cannot simplify
        return None

    def visit_DeclStmt(self, node: DeclStmt) -> None:
        """Analyze typed declaration statement."""
        # Try to simplify signal literal projections before type checking
        # This transforms: Signal x = 50 | "signal-A"
        # Into: Signal x = ("signal-A", 50)
        if node.type_name == "Signal" and isinstance(node.value, ProjectionExpr):
            simplified = self._try_simplify_signal_projection(node.value)
            if simplified is not None:
                node.value = simplified

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
                stage="semantic",
                node=node,
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
            self.diagnostics.error(e.message, stage="semantic", node=node)
            return

        self._register_signal_metadata(node.name, node, value_type, node.type_name)
        self._track_source_nature(node.name, node.value)

    def _type_name_to_symbol_type(self, type_name: str) -> SymbolType:
        """Convert type name to symbol type."""
        if type_name == "Entity":
            return SymbolType.ENTITY
        elif type_name == "Memory":
            return SymbolType.MEMORY
        else:
            return SymbolType.VARIABLE

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

    def _get_function_return_type(self, function_name: str) -> ValueInfo:
        """Determine the return type of a function by analyzing its return statements."""

        func_symbol = self.current_scope.lookup(function_name)
        if not func_symbol or func_symbol.symbol_type != SymbolType.FUNCTION:
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

        if isinstance(expr, IdentifierExpr):
            # Check if this identifier is a parameter
            symbol = self.current_scope.lookup(expr.name)
            return symbol and symbol.symbol_type == SymbolType.PARAMETER
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
        return symbol and symbol.symbol_type == SymbolType.PARAMETER

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
            self._validate_place_call(node)
        elif node.name == "memory":
            self._validate_memory_call(node)
        else:
            # Generic fallback - analyze all arguments
            for arg in node.args:
                self.get_expr_type(arg)

    def _validate_place_call(self, node: CallExpr) -> None:
        """Validate place() builtin call."""
        if not (3 <= len(node.args) <= 4):
            self.diagnostics.error(
                "place() requires 3 or 4 arguments (prototype, x, y, [properties])",
                stage="semantic",
                node=node,
            )
            return

        prototype = node.args[0]
        if not isinstance(prototype, StringLiteral):
            self.diagnostics.error(
                "place() prototype must be a string literal",
                stage="semantic",
                node=(prototype if node.args else node),
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
                    stage="semantic",
                    node=node.args[3],
                )
            else:
                # Analyze property values
                for value in node.args[3].entries.values():
                    self.get_expr_type(value)

        if isinstance(prototype, StringLiteral):
            node.metadata["prototype"] = prototype.value

    def _validate_memory_call(self, node: CallExpr) -> None:
        """Validate memory() builtin call."""
        if not (1 <= len(node.args) <= 2):
            self.diagnostics.error(
                "memory() expects one or two arguments (initial, [type])",
                stage="semantic",
                node=node,
            )

        if len(node.args) >= 1:
            self.get_expr_type(node.args[0])

        if len(node.args) == 2 and not isinstance(node.args[1], StringLiteral):
            self.diagnostics.error(
                "memory() explicit type must be a string literal",
                stage="semantic",
                node=node.args[1],
            )
        elif len(node.args) == 2:
            node.metadata["explicit_type"] = node.args[1].value

    def visit_MemDecl(self, node: MemDecl) -> None:
        """Analyze memory declaration."""
        declared_type = node.signal_type

        if declared_type is None:
            # Allocate a compiler-managed virtual channel placeholder. The
            # concrete storage channel will be inferred from the first write().
            signal_info = self.allocate_implicit_type()
        else:
            # Validate the signal type exists in Factorio
            self.validate_signal_type_with_error(
                declared_type, node, f"for memory '{node.name}'"
            )
            if declared_type in self.RESERVED_SIGNAL_RULES:
                self._emit_reserved_signal_diagnostic(
                    declared_type, node, "for memory storage"
                )

            signal_info = self.make_signal_type_info(declared_type)

        memory_type = SignalValue(signal_type=signal_info)

        symbol = Symbol(
            name=node.name,
            symbol_type=SymbolType.MEMORY,
            value_type=memory_type,
            defined_at=node,
            is_mutable=True,
        )
        try:
            self.current_scope.define(symbol)
        except SemanticError as e:
            self.diagnostics.error(e.message, stage="semantic", node=node)
            return

        mem_info = MemoryInfo(
            name=node.name,
            symbol=symbol,
            signal_type=declared_type,
            signal_info=signal_info,
            explicit=declared_type is not None,
        )
        self.memory_types[node.name] = mem_info

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
            symbol_type=SymbolType.FUNCTION,
            value_type=FunctionValue(param_types=param_types, return_type=return_type),
            defined_at=node,
            function_def=node,  # Store AST for analysis
        )
        try:
            self.current_scope.define(func_symbol)
        except SemanticError as e:
            self.diagnostics.error(e.message, stage="semantic", node=node)
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
                symbol_type=SymbolType.PARAMETER,
                value_type=param_type,
                defined_at=node,
            )
            try:
                self.current_scope.define(param_symbol)
            except SemanticError as e:
                self.diagnostics.error(e.message, stage="semantic", node=node)

        # Analyze function body
        for stmt in node.body:
            self.visit(stmt)

        # Restore scope
        self.current_scope = old_scope

    def _infer_function_return_type(self, body: List[Statement]) -> ValueInfo:
        """Infer function return type by analyzing return statements."""

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
                        symbol_type=SymbolType.ENTITY,
                        value_type=IntValue(),  # Entities don't have specific types
                        defined_at=node.target,
                        is_mutable=True,
                    )
                    try:
                        self.current_scope.define(entity_symbol)
                    except SemanticError as e:
                        self.diagnostics.error(
                            e.message, stage="semantic", node=node.target
                        )
                else:
                    self.diagnostics.error(
                        f"Undefined variable '{node.target.name}'",
                        stage="semantic",
                        node=node.target,
                    )
            elif not symbol.is_mutable and symbol.symbol_type != SymbolType.ENTITY:
                self.diagnostics.error(
                    f"Cannot assign to immutable '{node.target.name}'",
                    stage="semantic",
                    node=node.target,
                )
            else:
                target_symbol = symbol
        elif isinstance(node.target, PropertyAccess):
            # Check that the object exists
            object_symbol = self.current_scope.lookup(node.target.object_name)
            if object_symbol is None:
                self.diagnostics.error(
                    f"Undefined entity '{node.target.object_name}'",
                    stage="semantic",
                    node=node.target,
                )
            elif object_symbol.symbol_type != SymbolType.ENTITY:
                self.diagnostics.error(
                    f"Cannot access property '{node.target.property_name}' on non-entity '{node.target.object_name}'",
                    stage="semantic",
                    node=node.target,
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
            stage="semantic",
            node=node,
        )

    def visit_ReturnStmt(self, node: ReturnStmt) -> None:
        """Analyze return statement."""
        # Type check the return expression
        if node.expr:
            self.get_expr_type(node.expr)
            # Could check against function return type here

    def visit_CallExpr(self, node: CallExpr) -> None:
        """Analyze function call expressions."""
        builtin_names = {"place", "input", "memory"}

        func_symbol = self.current_scope.lookup(node.name)
        if func_symbol is None:
            if node.name in builtin_names:
                self._validate_builtin_call(node)
                return
            self.diagnostics.error(
                f"Undefined function '{node.name}'", stage="semantic", node=node
            )
            return

        if func_symbol.symbol_type != SymbolType.FUNCTION:
            self.diagnostics.error(
                f"'{node.name}' is not a function", stage="semantic", node=node
            )
            return

        # User-defined function call
        if hasattr(func_symbol, "function_def") and func_symbol.function_def:
            expected_params = len(func_symbol.function_def.params)
            actual_args = len(node.args)
            if actual_args != expected_params:
                self.diagnostics.error(
                    f"Function '{node.name}' expects {expected_params} arguments, got {actual_args}",
                    stage="semantic",
                    node=node,
                )

        for arg in node.args:
            self.get_expr_type(arg)

    def generic_visit(self, node: ASTNode) -> Any:
        """Default visitor - traverse children."""
        for field_name, field_value in node.__dict__.items():
            if isinstance(field_value, ASTNode):
                self.visit(field_value)
            elif isinstance(field_value, list):
                for item in field_value:
                    if isinstance(item, ASTNode):
                        self.visit(item)
