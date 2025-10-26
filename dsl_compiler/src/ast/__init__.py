"""AST node definitions for the Factorio Circuit DSL."""

from .base import ASTNode, ASTVisitor, ASTTransformer, ast_to_dict, print_ast
from .expressions import (
    Expr,
    BinaryOp,
    UnaryOp,
    CallExpr,
    ReadExpr,
    WriteExpr,
    ProjectionExpr,
    SignalLiteral,
    IdentifierExpr,
    PropertyAccessExpr,
    PlaceExpr,
)
from .statements import (
    Statement,
    DeclStmt,
    AssignStmt,
    MemDecl,
    ExprStmt,
    ReturnStmt,
    ImportStmt,
    FuncDecl,
    Program,
    TopDecl,
)
from .literals import (
    Literal,
    NumberLiteral,
    StringLiteral,
    DictLiteral,
    LValue,
    Identifier,
    PropertyAccess,
)
from .types import TypeInfo, SignalType, IntType

__all__ = [
    # Base classes
    "ASTNode",
    "ASTVisitor",
    "ASTTransformer",
    "ast_to_dict",
    "print_ast",
    # Expressions
    "Expr",
    "BinaryOp",
    "UnaryOp",
    "CallExpr",
    "ReadExpr",
    "WriteExpr",
    "ProjectionExpr",
    "SignalLiteral",
    "IdentifierExpr",
    "PropertyAccessExpr",
    "PlaceExpr",
    # Statements
    "Statement",
    "DeclStmt",
    "AssignStmt",
    "MemDecl",
    "ExprStmt",
    "ReturnStmt",
    "ImportStmt",
    "FuncDecl",
    "Program",
    "TopDecl",
    # Literals
    "Literal",
    "NumberLiteral",
    "StringLiteral",
    "DictLiteral",
    "LValue",
    "Identifier",
    "PropertyAccess",
    # Types
    "TypeInfo",
    "SignalType",
    "IntType",
]
