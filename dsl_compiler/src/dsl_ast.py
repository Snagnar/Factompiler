# ast.py
"""
Abstract Syntax Tree node classes for the Factorio Circuit DSL.

These classes represent the parsed structure of DSL programs and provide
a foundation for semantic analysis and IR generation.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union, Any, Dict


# =============================================================================
# Base AST Node
# =============================================================================


class ASTNode(ABC):
    """Base class for all AST nodes."""

    def __init__(self, line: int = 0, column: int = 0):
        self.line = line
        self.column = column


# =============================================================================
# Top-level Program Structure
# =============================================================================


class Program(ASTNode):
    """Root node representing an entire DSL program."""

    def __init__(self, statements: List["Statement"], line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.statements = statements


class TopDecl(ASTNode):
    """Base class for top-level declarations."""

    def __init__(self, line: int = 0, column: int = 0):
        super().__init__(line, column)


# =============================================================================
# Statements
# =============================================================================


class Statement(ASTNode):
    """Base class for all statements."""

    def __init__(self, line: int = 0, column: int = 0):
        super().__init__(line, column)


class DeclStmt(Statement):
    """type variable = expression;"""

    def __init__(
        self, type_name: str, name: str, value: "Expr", line: int = 0, column: int = 0
    ):
        super().__init__(line, column)
        self.type_name = type_name  # "int", "Signal", "SignalType", "Entity", "Bundle"
        self.name = name
        self.value = value


class AssignStmt(Statement):
    """variable = expression; or variable.property = expression;"""

    def __init__(self, target: "LValue", value: "Expr", line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.target = target
        self.value = value


class MemDecl(Statement):
    """mem name = memory([init_expr]);"""

    def __init__(
        self,
        name: str,
        init_expr: Optional["Expr"] = None,
        line: int = 0,
        column: int = 0,
    ):
        super().__init__(line, column)
        self.name = name
        self.init_expr = init_expr


class ExprStmt(Statement):
    """expression; (standalone expression statement)"""

    def __init__(self, expr: "Expr", line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.expr = expr


class ReturnStmt(Statement):
    """return expression;"""

    def __init__(self, expr: "Expr", line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.expr = expr


class ImportStmt(Statement):
    """import "file" [as alias];"""

    def __init__(
        self, path: str, alias: Optional[str] = None, line: int = 0, column: int = 0
    ):
        super().__init__(line, column)
        self.path = path
        self.alias = alias


class FuncDecl(TopDecl):
    """func name(params...) { statements... }"""

    def __init__(
        self,
        name: str,
        params: List[str],
        body: List[Statement],
        line: int = 0,
        column: int = 0,
    ):
        super().__init__(line, column)
        self.name = name
        self.params = params
        self.body = body


# =============================================================================
# L-Values (assignment targets)
# =============================================================================


class LValue(ASTNode):
    """Base class for assignment targets."""

    def __init__(self, line: int = 0, column: int = 0):
        super().__init__(line, column)


class Identifier(LValue):
    """Simple variable reference."""

    def __init__(self, name: str, line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.name = name


class PropertyAccess(LValue):
    """entity.property reference."""

    def __init__(
        self, object_name: str, property_name: str, line: int = 0, column: int = 0
    ):
        super().__init__(line, column)
        self.object_name = object_name
        self.property_name = property_name


# =============================================================================
# Expressions
# =============================================================================


class Expr(ASTNode):
    """Base class for all expressions."""

    def __init__(self, line: int = 0, column: int = 0):
        super().__init__(line, column)


class BinaryOp(Expr):
    """Binary operation: left op right"""

    def __init__(
        self, op: str, left: Expr, right: Expr, line: int = 0, column: int = 0
    ):
        super().__init__(line, column)
        self.op = op  # +, -, *, /, %, ==, !=, <, <=, >, >=, &&, ||
        self.left = left
        self.right = right


class UnaryOp(Expr):
    """Unary operation: op expr"""

    def __init__(self, op: str, expr: Expr, line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.op = op  # +, -, !
        self.expr = expr


class Literal(Expr):
    """Base class for literal values."""

    def __init__(self, line: int = 0, column: int = 0):
        super().__init__(line, column)


class NumberLiteral(Literal):
    """Numeric literal: 42, -17"""

    def __init__(self, value: int, line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.value = value


class StringLiteral(Literal):
    """String literal: "iron-plate" """

    def __init__(self, value: str, line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.value = value


class IdentifierExpr(Expr):
    """Variable reference in expression context."""

    def __init__(self, name: str, line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.name = name


class PropertyAccessExpr(Expr):
    """Property access in expression context: entity.property"""

    def __init__(
        self, object_name: str, property_name: str, line: int = 0, column: int = 0
    ):
        super().__init__(line, column)
        self.object_name = object_name
        self.property_name = property_name


class CallExpr(Expr):
    """Function call: func(args...)"""

    def __init__(self, name: str, args: List[Expr], line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.name = name
        self.args = args


class InputExpr(Expr):
    """input(index) or input(type, index)"""

    def __init__(
        self,
        index: Expr,
        signal_type: Optional[Expr] = None,
        line: int = 0,
        column: int = 0,
    ):
        super().__init__(line, column)
        self.index = index
        self.signal_type = signal_type  # if present, this is input(type, index)


class ReadExpr(Expr):
    """read(memory_name)"""

    def __init__(self, memory_name: str, line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.memory_name = memory_name


class WriteExpr(Expr):
    """write(memory_name, value)"""

    def __init__(self, memory_name: str, value: Expr, line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.memory_name = memory_name
        self.value = value


class BundleExpr(Expr):
    """bundle(expr1, expr2, ...)"""

    def __init__(self, exprs: List[Expr], line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.exprs = exprs


class ProjectionExpr(Expr):
    """expr | "type" - project signal/bundle to specific channel"""

    def __init__(self, expr: Expr, target_type: str, line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.expr = expr
        self.target_type = target_type  # the type literal after |


class PlaceExpr(Expr):
    """Place(proto, x, y, [props]) - entity placement"""

    def __init__(
        self,
        prototype: Expr,
        x: Expr,
        y: Expr,
        properties: Optional[Dict[str, Any]] = None,
        line: int = 0,
        column: int = 0,
    ):
        super().__init__(line, column)
        self.prototype = prototype  # string or identifier
        self.x = x
        self.y = y
        self.properties = properties  # For future extension


# =============================================================================
# Type Information (for semantic analysis)
# =============================================================================


class TypeInfo:
    """Type information attached to expressions during semantic analysis."""

    pass


class SignalType(TypeInfo):
    """Single-channel signal type: (type_name, count)"""

    def __init__(self, type_name: str, is_implicit: bool = False):
        self.type_name = type_name  # e.g. "iron-plate", "signal-A", "__v1"
        self.is_implicit = is_implicit  # True for compiler-allocated virtual signals


class BundleType(TypeInfo):
    """Multi-channel bundle type: Map[type_name -> count]"""

    def __init__(self, channels: Dict[str, "SignalType"]):
        self.channels = channels


class IntType(TypeInfo):
    """Plain integer type (not a signal)"""

    pass


# =============================================================================
# AST Visitor Pattern
# =============================================================================


class ASTVisitor(ABC):
    """Base class for AST traversal visitors."""

    @abstractmethod
    def visit(self, node: ASTNode) -> Any:
        """Visit a node and return result."""
        method_name = f"visit_{type(node).__name__}"
        if hasattr(self, method_name):
            return getattr(self, method_name)(node)
        else:
            return self.generic_visit(node)

    def generic_visit(self, node: ASTNode) -> Any:
        """Default visitor for unhandled node types."""
        pass


class ASTTransformer(ASTVisitor):
    """Base class for AST transformation visitors."""

    def visit(self, node: ASTNode) -> ASTNode:
        """Visit and transform a node."""
        method_name = f"visit_{type(node).__name__}"
        if hasattr(self, method_name):
            return getattr(self, method_name)(node)
        else:
            return self.generic_visit(node)

    def generic_visit(self, node: ASTNode) -> ASTNode:
        """Default transformer - returns node unchanged."""
        return node


# =============================================================================
# Utility Functions
# =============================================================================


def ast_to_dict(node: ASTNode) -> Dict:
    """Convert AST node to dictionary representation for debugging."""
    if not isinstance(node, ASTNode):
        return node

    result = {"type": type(node).__name__}
    for field_name, field_value in node.__dict__.items():
        if field_name in ("line", "column"):
            continue
        if isinstance(field_value, list):
            result[field_name] = [ast_to_dict(item) for item in field_value]
        elif isinstance(field_value, ASTNode):
            result[field_name] = ast_to_dict(field_value)
        else:
            result[field_name] = field_value

    return result


def print_ast(node: ASTNode, indent: int = 0) -> None:
    """Pretty-print AST structure for debugging."""
    spaces = "  " * indent
    print(f"{spaces}{type(node).__name__}")

    for field_name, field_value in node.__dict__.items():
        if field_name in ("line", "column"):
            continue
        print(f"{spaces}  {field_name}:", end="")
        if isinstance(field_value, ASTNode):
            print()
            print_ast(field_value, indent + 2)
        elif isinstance(field_value, list):
            print()
            for item in field_value:
                if isinstance(item, ASTNode):
                    print_ast(item, indent + 2)
                else:
                    print(f"{spaces}    {item}")
        else:
            print(f" {field_value}")
