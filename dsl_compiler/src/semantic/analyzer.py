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
    OutputSpecExpr,
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
    SignalTypeInfo,
    is_valid_factorio_signal,
)
from dsl_compiler.src.common.source_location import (
    SourceLocation,
)

from .symbol_table import SymbolTable, Symbol, SymbolType, SemanticError
from .type_system import (
    FunctionValue,
    IntValue,
    MemoryInfo,
    SignalDebugInfo,
    SignalValue,
    ValueInfo,
)


class SemanticAnalyzer(ASTVisitor):
    """Main semantic analysis visitor."""

    RESERVED_SIGNAL_RULES: Dict[str, Tuple[str, str]] = {
        "signal-W": ("error", "the memory write-enable channel"),
    }

    # Operator categories for type inference
    ARITHMETIC_OPS = {"+", "-", "*", "/", "%", "**"}
    BITWISE_OPS = {"<<", ">>", "AND", "OR", "XOR"}
    COMPARISON_OPS = {"==", "!=", "<", "<=", ">", ">="}
    LOGICAL_OPS = {"&&", "||"}

    def __init__(self, diagnostics: ProgramDiagnostics, strict_types: bool = False):
        self.strict_types = strict_types
        self.diagnostics = diagnostics
        self.diagnostics.default_stage = "semantic"
        self.symbol_table = SymbolTable()
        self.current_scope = self.symbol_table

        self.signal_registry = SignalTypeRegistry()

        self.signal_debug_info: Dict[str, SignalDebugInfo] = {}
        self.memory_types: Dict[str, MemoryInfo] = {}
        self.expr_types: Dict[int, ValueInfo] = {}

    @property
    def signal_type_map(self) -> Dict[str, Any]:
        """Get signal type map from registry."""
        return self.signal_registry.get_all_mappings()

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
        return self.signal_registry.allocate_implicit_type()

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

        value_type = self.infer_expr_type(expr)
        self.expr_types[expr_id] = value_type
        return value_type

    def infer_expr_type(self, expr: Expr) -> ValueInfo:
        """Infer the type of an expression."""
        if isinstance(expr, NumberLiteral):
            return IntValue(value=expr.value)

        elif isinstance(expr, StringLiteral):
            return IntValue()

        elif isinstance(expr, DictLiteral):
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

            if isinstance(value_type, IntValue):
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

                is_implicit_type = write_signal_name.startswith("__v")

                if is_implicit_type:
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
            return operand_type

        elif isinstance(expr, ProjectionExpr):
            # Recursively analyze the source expression to ensure all sub-expressions
            # (including identifiers) are resolved and cached in the current scope.
            # This is essential for function parameters to be found during lowering.
            self.get_expr_type(expr.expr)

            if expr.target_type in self.RESERVED_SIGNAL_RULES:
                self._emit_reserved_signal_diagnostic(
                    expr.target_type, expr, "as a projection target"
                )
            self.validate_signal_type_with_error(
                expr.target_type, expr, "in projection"
            )
            target_signal_type = SignalTypeInfo(name=expr.target_type)
            return SignalValue(signal_type=target_signal_type)

        elif isinstance(expr, SignalLiteral):
            # Recursively analyze the value expression to ensure all sub-expressions
            # are resolved and cached in the current scope.
            self.get_expr_type(expr.value)

            if expr.signal_type:
                if expr.signal_type in self.RESERVED_SIGNAL_RULES:
                    self._emit_reserved_signal_diagnostic(
                        expr.signal_type, expr, "in signal literals"
                    )
                self.validate_signal_type_with_error(
                    expr.signal_type, expr, "in signal literal"
                )
                signal_type = SignalTypeInfo(name=expr.signal_type)
                return SignalValue(signal_type=signal_type, count_expr=expr.value)
            else:
                if isinstance(expr.value, NumberLiteral):
                    return IntValue()
                else:
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
            object_symbol = self.current_scope.lookup(expr.object_name)
            if object_symbol is None:
                self.diagnostics.error(
                    f"Undefined variable '{expr.object_name}'",
                    stage="semantic",
                    node=expr,
                )
                return IntValue()
            elif object_symbol.symbol_type == SymbolType.ENTITY:
                return SignalValue(signal_type=self.allocate_implicit_type())
            elif object_symbol.symbol_type == SymbolType.MODULE:
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
            object_symbol = self.current_scope.lookup(expr.object_name)
            if object_symbol is None:
                self.diagnostics.error(
                    f"Undefined variable '{expr.object_name}'",
                    stage="semantic",
                    node=expr,
                )
                return IntValue()
            elif object_symbol.symbol_type == SymbolType.ENTITY:
                return SignalValue(signal_type=self.allocate_implicit_type())
            elif object_symbol.symbol_type == SymbolType.MODULE:
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

        elif isinstance(expr, OutputSpecExpr):
            return self._infer_output_spec_type(expr)

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
        if isinstance(left_type, IntValue) and isinstance(right_type, IntValue):
            return IntValue(), None

        if isinstance(left_type, SignalValue) and isinstance(right_type, IntValue):
            return left_type, None

        if isinstance(left_type, IntValue) and isinstance(right_type, SignalValue):
            return right_type, None

        if isinstance(left_type, SignalValue) and isinstance(right_type, SignalValue):
            if left_type.signal_type.name == right_type.signal_type.name:
                return left_type, None

            left_is_virtual = self._is_virtual_channel(left_type.signal_type)
            right_is_virtual = self._is_virtual_channel(right_type.signal_type)
            if left_is_virtual and right_is_virtual:
                return left_type, None

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

        return IntValue(), f"Invalid operand types for {op}"

    def infer_binary_op_type(self, expr: BinaryOp) -> ValueInfo:
        """Infer type for binary operations with mixed-type rules."""
        left_type = self.get_expr_type(expr.left)
        right_type = self.get_expr_type(expr.right)

        # Comparison operators return a signal (for compatibility)
        if expr.op in self.COMPARISON_OPS:
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

        # Logical operators return a signal
        if expr.op in self.LOGICAL_OPS:
            if isinstance(left_type, SignalValue):
                return left_type
            if isinstance(right_type, SignalValue):
                return right_type
            return SignalValue(signal_type=self.allocate_implicit_type())

        # Bitwise operators follow same rules as arithmetic
        if expr.op in self.BITWISE_OPS:
            result_type, warning_msg = self._check_signal_type_compatibility(
                left_type, right_type, expr.op, expr
            )
            if warning_msg:
                if self.strict_types:
                    self.diagnostics.error(warning_msg, stage="semantic", node=expr)
                else:
                    self.diagnostics.warning(warning_msg, stage="semantic", node=expr)
            return result_type

        # Power operator (same as arithmetic)
        if expr.op == "**":
            result_type, warning_msg = self._check_signal_type_compatibility(
                left_type, right_type, expr.op, expr
            )
            if warning_msg:
                if self.strict_types:
                    self.diagnostics.error(warning_msg, stage="semantic", node=expr)
                else:
                    self.diagnostics.warning(warning_msg, stage="semantic", node=expr)
            return result_type

        # Standard arithmetic operators
        result_type, warning_msg = self._check_signal_type_compatibility(
            left_type, right_type, expr.op, expr
        )

        if warning_msg:
            if self.strict_types:
                self.diagnostics.error(warning_msg, stage="semantic", node=expr)
            else:
                self.diagnostics.warning(warning_msg, stage="semantic", node=expr)

        return result_type

    def _infer_output_spec_type(self, expr: OutputSpecExpr) -> ValueInfo:
        """Infer type for output specifier expression.

        The result type is determined by the output_value, not the condition.
        """
        # Validate that condition is a comparison
        if not self._is_comparison_expr(expr.condition):
            self.diagnostics.error(
                "Output specifier (:) requires a comparison expression on the left. "
                f"Got: {type(expr.condition).__name__}",
                stage="semantic",
                node=expr,
            )

        # Analyze both parts
        self.get_expr_type(expr.condition)
        output_type = self.get_expr_type(expr.output_value)

        # Result type comes from output_value
        if isinstance(output_type, IntValue):
            # Integer constant output - result needs a signal type
            # Use the left operand's type from the comparison if available
            condition_type = self._get_comparison_left_type(expr.condition)
            if isinstance(condition_type, SignalValue):
                return condition_type
            else:
                return SignalValue(signal_type=self.allocate_implicit_type())

        return output_type

    def _is_comparison_expr(self, expr: Expr) -> bool:
        """Check if expression is a comparison operation."""
        if isinstance(expr, BinaryOp):
            return expr.op in self.COMPARISON_OPS
        return False

    def _get_comparison_left_type(self, expr: Expr) -> ValueInfo:
        """Get the type of the left operand in a comparison."""
        if isinstance(expr, BinaryOp) and expr.op in self.COMPARISON_OPS:
            return self.get_expr_type(expr.left)
        return IntValue()

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

        return None

    def visit_DeclStmt(self, node: DeclStmt) -> None:
        """Analyze typed declaration statement."""
        if node.type_name == "Signal" and isinstance(node.value, ProjectionExpr):
            simplified = self._try_simplify_signal_projection(node.value)
            if simplified is not None:
                node.value = simplified

        value_type = self.get_expr_type(node.value)

        if node.type_name == "Signal" and isinstance(value_type, IntValue):
            implicit_signal_type = self.allocate_implicit_type()
            value_type = SignalValue(
                signal_type=implicit_signal_type,
                count_expr=node.value,
            )

        symbol_type = self._type_name_to_symbol_type(node.type_name)

        if not self._value_matches_type(value_type, node.type_name):
            self.diagnostics.error(
                f"Cannot assign {self._value_type_name(value_type)} to {node.type_name} variable '{node.name}'",
                stage="semantic",
                node=node,
            )

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

        if hasattr(func_symbol, "function_def") and func_symbol.function_def:
            for stmt in func_symbol.function_def.body:
                if isinstance(stmt, ReturnStmt) and stmt.expr:
                    return_type = self.get_expr_type(stmt.expr)
                    return return_type

        return SignalValue(signal_type=self.allocate_implicit_type())

    def _infer_parameter_type(self, param_name: str, func_def) -> ValueInfo:
        """Infer parameter type from usage within the function body."""
        return SignalValue(signal_type=self.allocate_implicit_type())

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
            signal_info = self.allocate_implicit_type()
        else:
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
        param_types = []
        for param_name in node.params:
            param_types.append(IntValue())

        return_type = self._infer_function_return_type(node.body)

        func_symbol = Symbol(
            name=node.name,
            symbol_type=SymbolType.FUNCTION,
            value_type=FunctionValue(param_types=param_types, return_type=return_type),
            defined_at=node,
            function_def=node,
        )
        try:
            self.current_scope.define(func_symbol)
        except SemanticError as e:
            self.diagnostics.error(e.message, stage="semantic", node=node)
            return

        func_scope = self.current_scope.create_child_scope()
        old_scope = self.current_scope
        self.current_scope = func_scope

        for param_name in node.params:
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

        for stmt in node.body:
            self.visit(stmt)

        self.current_scope = old_scope

    def _infer_function_return_type(self, body: List[Statement]) -> ValueInfo:
        """Infer function return type by analyzing return statements."""

        for stmt in body:
            if isinstance(stmt, ReturnStmt) and stmt.expr:
                return IntValue()

        return IntValue()

    def visit_ExprStmt(self, node: ExprStmt) -> None:
        """Analyze expression statement."""
        self.get_expr_type(node.expr)

    def visit_AssignStmt(self, node: AssignStmt) -> None:
        """Analyze assignment statement."""
        target_symbol = None
        if isinstance(node.target, Identifier):
            symbol = self.current_scope.lookup(node.target.name)
            if symbol is None:
                if isinstance(node.value, CallExpr) and node.value.name == "place":
                    entity_symbol = Symbol(
                        name=node.target.name,
                        symbol_type=SymbolType.ENTITY,
                        value_type=IntValue(),
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

        value_type = self.get_expr_type(node.value)

        if isinstance(node.target, Identifier):
            self._register_signal_metadata(
                node.target.name,
                node.target,
                value_type,
                target_symbol.symbol_type if target_symbol else None,
            )

    def visit_ImportStmt(self, node: ImportStmt) -> None:
        """Analyze import statement."""
        self.diagnostics.error(
            f"Import statement found in AST - file may not have been found during preprocessing: {node.path}",
            stage="semantic",
            node=node,
        )

    def visit_ReturnStmt(self, node: ReturnStmt) -> None:
        """Analyze return statement."""
        if node.expr:
            self.get_expr_type(node.expr)

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
