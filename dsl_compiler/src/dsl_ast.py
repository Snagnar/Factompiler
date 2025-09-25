# ast.py
"""
Abstract Syntax Tree node classes for the Factorio Circuit DSL.

These classes represent the parsed structure of DSL programs and provide
a foundation for semantic analysis and IR generation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
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

@dataclass
class Program(ASTNode):
    """Root node representing an entire DSL program."""
    statements: List['Statement']
    
    def __init__(self, statements: List['Statement'], line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.statements = statements


@dataclass  
class TopDecl(ASTNode):
    """Base class for top-level declarations."""
    def __init__(self, line: int = 0, column: int = 0):
        super().__init__(line, column)


# =============================================================================
# Statements
# =============================================================================

@dataclass
class Statement(ASTNode):
    """Base class for all statements."""
    def __init__(self, line: int = 0, column: int = 0):
        super().__init__(line, column)


@dataclass
class LetStmt(Statement):
    """let variable = expression;"""
    name: str
    value: 'Expr'
    
    def __init__(self, name: str, value: 'Expr', line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.name = name
        self.value = value


@dataclass
class AssignStmt(Statement):
    """variable = expression; or variable.property = expression;"""
    target: 'LValue'
    value: 'Expr'
    
    def __init__(self, target: 'LValue', value: 'Expr', line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.target = target
        self.value = value


@dataclass
class MemDecl(Statement):
    """mem name = memory([init_expr]);"""
    name: str
    init_expr: Optional['Expr']
    
    def __init__(self, name: str, init_expr: Optional['Expr'] = None, line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.name = name
        self.init_expr = init_expr


@dataclass
class ExprStmt(Statement):
    """expression; (standalone expression statement)"""
    expr: 'Expr'
    
    def __init__(self, expr: 'Expr', line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.expr = expr


@dataclass
class ReturnStmt(Statement):
    """return expression;"""
    expr: 'Expr'
    
    def __init__(self, expr: 'Expr', line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.expr = expr


@dataclass
class ImportStmt(Statement):
    """import "file" [as alias];"""
    path: str
    alias: Optional[str] = None
    
    def __init__(self, path: str, alias: Optional[str] = None, line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.path = path
        self.alias = alias


@dataclass
class FuncDecl(TopDecl):
    """func name(params...) { statements... }"""
    name: str
    params: List[str]
    body: List[Statement]
    
    def __init__(self, name: str, params: List[str], body: List[Statement], line: int = 0, column: int = 0):
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


@dataclass
class Identifier(LValue):
    """Simple variable reference."""
    name: str
    
    def __init__(self, name: str, line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.name = name


@dataclass
class PropertyAccess(LValue):
    """entity.property reference."""
    object_name: str
    property_name: str
    
    def __init__(self, object_name: str, property_name: str, line: int = 0, column: int = 0):
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


@dataclass
class BinaryOp(Expr):
    """Binary operation: left op right"""
    op: str  # +, -, *, /, %, ==, !=, <, <=, >, >=, &&, ||
    left: Expr
    right: Expr
    
    def __init__(self, op: str, left: Expr, right: Expr, line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.op = op
        self.left = left
        self.right = right


@dataclass
class UnaryOp(Expr):
    """Unary operation: op expr"""
    op: str  # +, -, !
    expr: Expr
    
    def __init__(self, op: str, expr: Expr, line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.op = op
        self.expr = expr


class Literal(Expr):
    """Base class for literal values."""
    def __init__(self, line: int = 0, column: int = 0):
        super().__init__(line, column)


@dataclass
class NumberLiteral(Literal):
    """Numeric literal: 42, -17"""
    value: int
    
    def __init__(self, value: int, line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.value = value


@dataclass
class StringLiteral(Literal):
    """String literal: "iron-plate" """
    value: str
    
    def __init__(self, value: str, line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.value = value


@dataclass
class IdentifierExpr(Expr):
    """Variable reference in expression context."""
    name: str
    
    def __init__(self, name: str, line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.name = name


@dataclass
class PropertyAccessExpr(Expr):
    """Property access in expression context: entity.property"""
    object_name: str
    property_name: str
    
    def __init__(self, object_name: str, property_name: str, line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.object_name = object_name
        self.property_name = property_name


@dataclass
class CallExpr(Expr):
    """Function call: func(args...)"""
    name: str
    args: List[Expr]
    
    def __init__(self, name: str, args: List[Expr], line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.name = name
        self.args = args


@dataclass
class InputExpr(Expr):
    """input(index) or input(type, index)"""
    index: Expr
    signal_type: Optional[Expr] = None  # if present, this is input(type, index)
    
    def __init__(self, index: Expr, signal_type: Optional[Expr] = None, line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.index = index
        self.signal_type = signal_type


@dataclass
class ReadExpr(Expr):
    """read(memory_name)"""
    memory_name: str
    
    def __init__(self, memory_name: str, line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.memory_name = memory_name


@dataclass
class WriteExpr(Expr):
    """write(memory_name, value)"""
    memory_name: str
    value: Expr
    
    def __init__(self, memory_name: str, value: Expr, line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.memory_name = memory_name
        self.value = value


@dataclass
class BundleExpr(Expr):
    """bundle(expr1, expr2, ...)"""
    exprs: List[Expr]
    
    def __init__(self, exprs: List[Expr], line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.exprs = exprs


@dataclass
class ProjectionExpr(Expr):
    """expr | "type" - project signal/bundle to specific channel"""
    expr: Expr
    target_type: str  # the type literal after |
    
    def __init__(self, expr: Expr, target_type: str, line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.expr = expr
        self.target_type = target_type


@dataclass
class PlaceExpr(Expr):
    """Place(proto, x, y, [props]) - entity placement"""
    prototype: Expr  # string or identifier
    x: Expr
    y: Expr
    properties: Optional[Dict[str, Any]] = None  # For future extension
    
    def __init__(self, prototype: Expr, x: Expr, y: Expr, properties: Optional[Dict[str, Any]] = None, line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.prototype = prototype
        self.x = x
        self.y = y
        self.properties = properties


# =============================================================================
# Type Information (for semantic analysis)
# =============================================================================

@dataclass
class TypeInfo:
    """Type information attached to expressions during semantic analysis."""
    pass


@dataclass
class SignalType(TypeInfo):
    """Single-channel signal type: (type_name, count)"""
    type_name: str  # e.g. "iron-plate", "signal-A", "__v1"
    is_implicit: bool = False  # True for compiler-allocated virtual signals


@dataclass
class BundleType(TypeInfo):
    """Multi-channel bundle type: Map[type_name -> count]"""
    channels: Dict[str, 'SignalType']


@dataclass
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
        method_name = f'visit_{type(node).__name__}'
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
        method_name = f'visit_{type(node).__name__}'
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
    
    result = {'type': type(node).__name__}
    for field_name, field_value in node.__dict__.items():
        if field_name in ('line', 'column'):
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
        if field_name in ('line', 'column'):
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
