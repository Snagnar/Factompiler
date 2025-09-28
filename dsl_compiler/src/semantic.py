# semantic.py
"""
Semantic analysis for the Factorio Circuit DSL.

This module performs:
- Symbol table construction and name resolution
- Type inference with implicit signal type allocation
- Mixed-type arithmetic validation and warnings
- Diagnostic collection and error reporting
"""

from typing import Dict, List, Optional, Set, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from dsl_compiler.src.dsl_ast import *


# =============================================================================
# Type System
# =============================================================================


class ValueType(Enum):
    """Possible value types in the DSL."""

    INT = "int"
    SIGNAL = "signal"
    BUNDLE = "bundle"


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
class BundleValue:
    """A multi-channel bundle value."""

    channels: Dict[str, SignalValue]  # type_name -> SignalValue


@dataclass
class IntValue:
    """A plain integer value."""

    value: Optional[int] = None  # None for computed values


@dataclass
class FunctionValue:
    """Value type for functions"""

    param_types: List["ValueInfo"] = field(default_factory=list)
    return_type: "ValueInfo" = field(default_factory=lambda: IntValue())


ValueInfo = Union[SignalValue, BundleValue, IntValue, FunctionValue]


# =============================================================================
# Symbol Table
# =============================================================================


@dataclass
class Symbol:
    """Symbol table entry."""

    name: str
    symbol_type: str  # "variable", "memory", "function", "parameter", "entity", "module"
    value_type: ValueInfo
    defined_at: ASTNode
    is_mutable: bool = False
    properties: Optional[Dict[str, "Symbol"]] = None  # For modules and entities
    function_def: Optional[ASTNode] = None  # For functions - store AST for inlining


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

    def __init__(self, strict_types: bool = False):
        self.strict_types = strict_types
        self.diagnostics = DiagnosticCollector()
        self.symbol_table = SymbolTable()
        self.current_scope = self.symbol_table

        # Type allocation
        self.implicit_type_counter = 0
        self.signal_type_map: Dict[str, str] = {}  # implicit -> factorio signal

        # Expression type cache (using id() as key since AST nodes aren't hashable)
        self.expr_types: Dict[int, ValueInfo] = {}

    def allocate_implicit_type(self) -> SignalTypeInfo:
        """Allocate a new implicit virtual signal type."""
        self.implicit_type_counter += 1
        implicit_name = f"__v{self.implicit_type_counter}"
        factorio_signal = (
            f"signal-{chr(ord('A') + (self.implicit_type_counter - 1) % 26)}"
        )

        # Store mapping for debugging
        self.signal_type_map[implicit_name] = factorio_signal

        return SignalTypeInfo(name=implicit_name, is_implicit=True, is_virtual=True)

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

        elif isinstance(expr, IdentifierExpr):
            symbol = self.current_scope.lookup(expr.name)
            if symbol is None:
                self.diagnostics.error(f"Undefined variable '{expr.name}'", expr)
                return IntValue()
            return symbol.value_type

        elif isinstance(expr, InputExpr):
            # input(index) or input(type, index)
            if expr.signal_type is None:
                # Allocate implicit type
                signal_type = self.allocate_implicit_type()
                return SignalValue(signal_type=signal_type)
            else:
                # Extract type from signal_type expression
                if isinstance(expr.signal_type, StringLiteral):
                    type_name = expr.signal_type.value
                    signal_type = SignalTypeInfo(name=type_name)
                    return SignalValue(signal_type=signal_type)
                else:
                    self.diagnostics.error(
                        "Signal type must be string literal", expr.signal_type
                    )
                    return SignalValue(signal_type=self.allocate_implicit_type())

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
            # write() operations return void, but we model as signal for simplicity
            symbol = self.current_scope.lookup(expr.memory_name)
            if symbol and symbol.symbol_type == "memory":
                return symbol.value_type
            return SignalValue(signal_type=self.allocate_implicit_type())

        elif isinstance(expr, BinaryOp):
            return self.infer_binary_op_type(expr)

        elif isinstance(expr, UnaryOp):
            operand_type = self.get_expr_type(expr.expr)
            return operand_type  # Unary ops preserve type

        elif isinstance(expr, ProjectionExpr):
            # expr | "type" always returns Signal of specified type
            target_signal_type = SignalTypeInfo(name=expr.target_type)
            return SignalValue(signal_type=target_signal_type)

        elif isinstance(expr, BundleExpr):
            # bundle(exprs...) returns Bundle
            channels = {}
            for sub_expr in expr.exprs:
                sub_type = self.get_expr_type(sub_expr)
                if isinstance(sub_type, SignalValue):
                    channels[sub_type.signal_type.name] = sub_type
                elif isinstance(sub_type, BundleValue):
                    # Merge bundle
                    channels.update(sub_type.channels)
                # Ignore IntValue in bundles
            return BundleValue(channels=channels)

        elif isinstance(expr, CallExpr):
            # Function calls - check if function returns entity
            if expr.name == "Place":
                # Direct Place() call returns entity type
                return SignalValue(
                    signal_type=self.allocate_implicit_type()
                )  # For now, keep as signal
            else:
                # Check if this is a user-defined function
                func_symbol = self.current_scope.lookup(expr.name)
                if func_symbol and func_symbol.symbol_type == "function":
                    # Analyze function return type
                    return self._get_function_return_type(expr.name)
                else:
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
            self.diagnostics.warning(f"Unknown expression type: {type(expr)}", expr)
            return IntValue()

    def infer_binary_op_type(self, expr: BinaryOp) -> ValueInfo:
        """Infer type for binary operations with mixed-type rules."""
        left_type = self.get_expr_type(expr.left)
        right_type = self.get_expr_type(expr.right)

        # Comparison operators always return int (0/1)
        if expr.op in ["==", "!=", "<", "<=", ">", ">=", "&&", "||"]:
            return IntValue()

        # Arithmetic operators: +, -, *, /, %
        if isinstance(left_type, IntValue) and isinstance(right_type, IntValue):
            # Int + Int = Int
            return IntValue()

        elif isinstance(left_type, SignalValue) and isinstance(right_type, IntValue):
            # Signal + Int = Signal (int coerced to signal's type)
            warning_msg = (
                f"Mixed types in binary operation: "
                f"signal '{left_type.signal_type.name}' {expr.op} integer. "
                f"Integer will be coerced to signal type."
            )
            if self.strict_types:
                self.diagnostics.error(warning_msg, expr)
            else:
                self.diagnostics.warning(warning_msg, expr)
            return left_type

        elif isinstance(left_type, IntValue) and isinstance(right_type, SignalValue):
            # Int + Signal = Signal (int coerced to signal's type)
            warning_msg = (
                f"Mixed types in binary operation: "
                f"integer {expr.op} signal '{right_type.signal_type.name}'. "
                f"Integer will be coerced to signal type."
            )
            if self.strict_types:
                self.diagnostics.error(warning_msg, expr)
            else:
                self.diagnostics.warning(warning_msg, expr)
            return right_type

        elif isinstance(left_type, SignalValue) and isinstance(right_type, SignalValue):
            # Signal + Signal
            if left_type.signal_type.name == right_type.signal_type.name:
                # Same type - can wire-merge or compute
                return left_type
            else:
                # Mixed types - left operand wins (with warning)
                warning_msg = (
                    f"Mixed signal types in binary operation: "
                    f"'{left_type.signal_type.name}' {expr.op} '{right_type.signal_type.name}'. "
                    f"Result will be '{left_type.signal_type.name}'. "
                    f"Use '| \"type\"' to explicitly set output channel."
                )

                if self.strict_types:
                    self.diagnostics.error(warning_msg, expr)
                else:
                    self.diagnostics.warning(warning_msg, expr)

                return left_type

        elif isinstance(left_type, BundleValue) or isinstance(right_type, BundleValue):
            # Bundle operations - complex merging rules
            return self.infer_bundle_operation_type(left_type, right_type, expr)

        else:
            self.diagnostics.error(f"Invalid operand types for {expr.op}", expr)
            return IntValue()

    def infer_bundle_operation_type(
        self, left_type: ValueInfo, right_type: ValueInfo, expr: BinaryOp
    ) -> BundleValue:
        """Handle bundle operations."""
        result_channels = {}

        # Convert operands to bundles
        left_bundle = self.to_bundle(left_type)
        right_bundle = self.to_bundle(right_type)

        # Merge channels
        all_types = set(left_bundle.channels.keys()) | set(right_bundle.channels.keys())

        for type_name in all_types:
            left_signal = left_bundle.channels.get(type_name)
            right_signal = right_bundle.channels.get(type_name)

            if left_signal and right_signal:
                # Both have this channel - combine
                result_channels[type_name] = left_signal  # Simplified
            elif left_signal:
                result_channels[type_name] = left_signal
            elif right_signal:
                result_channels[type_name] = right_signal

        return BundleValue(channels=result_channels)

    def to_bundle(self, value_type: ValueInfo) -> BundleValue:
        """Convert a value type to a bundle."""
        if isinstance(value_type, BundleValue):
            return value_type
        elif isinstance(value_type, SignalValue):
            return BundleValue(channels={value_type.signal_type.name: value_type})
        else:
            # IntValue -> empty bundle
            return BundleValue(channels={})

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
            # Check if this is a parameter coercion case in non-strict mode
            if (not self.strict_types and 
                node.type_name == "Bundle" and 
                self._expression_involves_parameter(node.value)):
                # In non-strict mode, allow parameter-based expressions to be coerced to Bundle
                # Issue a warning instead of an error
                self.diagnostics.warning(
                    f"Coercing {self._value_type_name(value_type)} to {node.type_name} for parameter-based expression in '{node.name}'",
                    node,
                )
                value_type = BundleValue(channels={})  # Treat as empty bundle for now
            else:
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

    def _type_name_to_symbol_type(self, type_name: str) -> str:
        """Convert type name to symbol type."""
        return "entity" if type_name == "Entity" else "variable"

    def _value_matches_type(
        self, value_type: ValueInfo, expected_type_name: str
    ) -> bool:
        """Check if a value type matches the expected type name."""
        type_map = {
            "int": IntValue,
            "Signal": SignalValue,
            "SignalType": SignalValue,  # For now, treat as Signal
            "Entity": SignalValue,  # Entity calls return signals for now
            "Bundle": BundleValue,
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
            BundleValue: "Bundle",
            FunctionValue: "function",
        }
        return type_names.get(type(value_type), "unknown")

    def _function_returns_entity(self, function_name: str) -> bool:
        """Check if a function returns an entity by examining its definition."""
        from dsl_compiler.src.dsl_ast import ReturnStmt, CallExpr

        func_symbol = self.current_scope.lookup(function_name)
        if not func_symbol or func_symbol.symbol_type != "function":
            return False

        # If the function definition has return statements with Place() calls
        if hasattr(func_symbol, "function_def") and func_symbol.function_def:
            for stmt in func_symbol.function_def.body:
                if isinstance(stmt, ReturnStmt) and stmt.expr:
                    if isinstance(stmt.expr, CallExpr) and stmt.expr.name == "Place":
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
        from dsl_compiler.src.dsl_ast import BinaryOp, DeclStmt, Identifier

        # Look for usage patterns in the function body
        for stmt in func_def.body:
            # Check for Bundle declarations that use this parameter
            if isinstance(stmt, DeclStmt) and stmt.type_name == "Bundle":
                if self._expression_uses_identifier(stmt.value, param_name):
                    # If a parameter is used in a Bundle assignment, it's likely a Bundle
                    return BundleValue(channels={})
            
            # Check for binary operations involving this parameter
            if isinstance(stmt, DeclStmt) and isinstance(stmt.value, BinaryOp):
                if self._expression_uses_identifier(stmt.value, param_name):
                    # If Bundle * param or param * Bundle, param is likely a scalar
                    if stmt.type_name == "Bundle":
                        # This suggests param is used in Bundle operations
                        return BundleValue(channels={})

        # Default to Signal type
        return SignalValue(signal_type=self.allocate_implicit_type())

    def _expression_uses_identifier(self, expr, identifier_name: str) -> bool:
        """Check if an expression uses a specific identifier."""
        from dsl_compiler.src.dsl_ast import IdentifierExpr, BinaryOp, CallExpr, PropertyAccess

        if isinstance(expr, IdentifierExpr):
            return expr.name == identifier_name
        elif isinstance(expr, BinaryOp):
            return (self._expression_uses_identifier(expr.left, identifier_name) or 
                    self._expression_uses_identifier(expr.right, identifier_name))
        elif isinstance(expr, CallExpr):
            return any(self._expression_uses_identifier(arg, identifier_name) for arg in expr.args)
        elif isinstance(expr, PropertyAccess):
            return expr.object_name == identifier_name
        # Add more expression types as needed
        return False

    def _expression_involves_parameter(self, expr) -> bool:
        """Check if an expression involves any function parameters."""
        from dsl_compiler.src.dsl_ast import IdentifierExpr, BinaryOp, CallExpr, PropertyAccess

        if isinstance(expr, IdentifierExpr):
            # Check if this identifier is a parameter
            symbol = self.current_scope.lookup(expr.name)
            return symbol and symbol.symbol_type == "parameter"
        elif isinstance(expr, BinaryOp):
            return (self._expression_involves_parameter(expr.left) or 
                    self._expression_involves_parameter(expr.right))
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
            return (self._expression_involves_parameter(expr.left) or 
                    self._expression_involves_parameter(expr.right))
        return False

    def visit_MemDecl(self, node: MemDecl) -> None:
        """Analyze memory declaration."""
        if node.init_expr:
            init_type = self.get_expr_type(node.init_expr)
            if isinstance(init_type, IntValue):
                # Convert int to signal with implicit type
                signal_type = self.allocate_implicit_type()
                memory_type = SignalValue(signal_type=signal_type)
            else:
                memory_type = init_type
        else:
            # Default initialization
            signal_type = self.allocate_implicit_type()
            memory_type = SignalValue(signal_type=signal_type)

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

    def visit_FuncDecl(self, node: FuncDecl) -> None:
        """Analyze function declaration."""
        # Create function symbol with placeholder type
        func_symbol = Symbol(
            name=node.name,
            symbol_type="function",
            value_type=IntValue(),  # Placeholder
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

    def visit_ExprStmt(self, node: ExprStmt) -> None:
        """Analyze expression statement."""
        self.get_expr_type(node.expr)

    def visit_AssignStmt(self, node: AssignStmt) -> None:
        """Analyze assignment statement."""
        # Check target exists and is mutable
        if isinstance(node.target, Identifier):
            symbol = self.current_scope.lookup(node.target.name)
            if symbol is None:
                # Check if this is entity assignment (Place() call)
                if isinstance(node.value, CallExpr) and node.value.name == "Place":
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

    def visit_ImportStmt(self, node: ImportStmt) -> None:
        """Analyze import statement."""
        # With C-style preprocessing, import statements should have been
        # replaced with the actual imported content, so this should rarely be called.
        # If we do encounter an import statement, it means the file wasn't found
        # during preprocessing, so we'll just log a warning.
        self.diagnostics.warning(
            f"Import statement found in AST - file may not have been found during preprocessing: {node.path}",
            node,
        )

    def visit_ReturnStmt(self, node: ReturnStmt) -> None:
        """Analyze return statement."""
        # Type check the return expression
        if node.expr:
            return_type = self.get_expr_type(node.expr)
            # Could check against function return type here

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
