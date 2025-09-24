# parser.py
"""
Parser module for the Factorio Circuit DSL.

Uses Lark parser with the grammar file to parse DSL source code into AST objects.
"""

import sys
from pathlib import Path
from typing import List, Optional, Any, Dict

from lark import Lark, Transformer, Tree, Token
from lark.exceptions import LarkError, ParseError, LexError

from dsl_ast import (
    ASTNode, Program, Statement, Expr, LValue,
    LetStmt, AssignStmt, MemDecl, ExprStmt, ReturnStmt, ImportStmt, FuncDecl,
    Identifier, PropertyAccess,
    BinaryOp, UnaryOp, NumberLiteral, StringLiteral, IdentifierExpr,
    CallExpr, InputExpr, ReadExpr, WriteExpr, BundleExpr, ProjectionExpr,
    PlaceExpr, print_ast
)


class DSLTransformer(Transformer):
    """Transforms Lark parse tree into typed AST nodes."""
    
    def __init__(self):
        super().__init__()
        self.line_info = {}  # Store line/column info
    
    def _set_position(self, node: ASTNode, token_or_tree) -> ASTNode:
        """Set line/column position on AST node from Lark token/tree."""
        if hasattr(token_or_tree, 'meta'):
            node.line = token_or_tree.meta.line
            node.column = token_or_tree.meta.column
        elif hasattr(token_or_tree, 'line'):
            node.line = token_or_tree.line
            node.column = token_or_tree.column
        return node
    
    # =========================================================================
    # Top-level constructs
    # =========================================================================
    
    def start(self, statements: List[Statement]) -> Program:
        """start: statement*"""
        return Program(statements=statements)
    
    def let_stmt(self, items) -> LetStmt:
        """let_stmt: "let" NAME "=" expr"""
        if len(items) == 1 and isinstance(items[0], LetStmt):
            # Handle nested let_stmt rule
            return items[0]
        elif len(items) >= 2:
            # Handle actual let statement tokens
            name = str(items[0])
            value = self._unwrap_tree(items[1])  # Handle Tree wrapper
            return self._set_position(LetStmt(name=name, value=value), items[0])
        else:
            raise ValueError(f"Unexpected let_stmt structure: {items}")
    
    def _unwrap_tree(self, item):
        """Unwrap Tree objects to get the actual value."""
        from lark import Tree
        if isinstance(item, Tree):
            if len(item.children) == 1:
                return self._unwrap_tree(item.children[0])
            else:
                return item.children
        return item
    
    def assign_stmt(self, items) -> AssignStmt:
        """assign_stmt: lvalue "=" expr"""
        # Check if we already have a complete AssignStmt (recursive transformation)
        if len(items) == 1 and isinstance(items[0], AssignStmt):
            stmt = items[0]
            # Extract the value from Tree if needed
            stmt.value = self._unwrap_tree(stmt.value)
            return stmt
            
        # Normal case: lvalue "=" expr
        if len(items) >= 2:
            target = items[0]
            value = self._unwrap_tree(items[1])
            return AssignStmt(target=target, value=value)
        else:
            raise ValueError(f"Could not parse assign_stmt from items: {items}")
    
    def mem_decl(self, items) -> MemDecl:
        """mem_decl: "mem" NAME "=" "memory" "(" [expr] ")" """
        # Handle case where we get a MemDecl back (from statement rule)
        if len(items) == 1 and isinstance(items[0], MemDecl):
            return items[0]
        
        # Parse the actual memory declaration
        # The structure should be: NAME, [expr] (tokens filtered by Lark)
        name = None
        init_expr = None
        
        for item in items:
            if isinstance(item, Token) and item.type == 'NAME':
                name = str(item)
            elif hasattr(item, '__class__') and 'Expr' in item.__class__.__name__:
                init_expr = item
            elif isinstance(item, Tree):
                # Unwrap tree to get expression
                init_expr = self._unwrap_tree(item)
        
        if name is None and len(items) > 0:
            # Fallback: assume first item is the name
            name = str(items[0])
            
        return self._set_position(MemDecl(name=name, init_expr=init_expr), items[0] if items else None)
    
    def expr_stmt(self, items) -> ExprStmt:
        """expr_stmt: expr"""
        return ExprStmt(expr=items[0])
    
    def return_stmt(self, items) -> ReturnStmt:
        """return_stmt: "return" expr"""
        return ReturnStmt(expr=items[0])
    
    def import_stmt(self, items) -> ImportStmt:
        """import_stmt: "import" STRING ["as" NAME]"""
        # If we already have an ImportStmt, return it
        if len(items) == 1 and isinstance(items[0], ImportStmt):
            return items[0]
            
        # Find the path string and optional alias
        path = None
        alias = None
        
        for i, item in enumerate(items):
            if isinstance(item, Token):
                if item.type == 'STRING':
                    path = item.value[1:-1]  # Remove quotes
                elif item.type == 'NAME' and path is not None:
                    alias = str(item)
            elif isinstance(item, str):
                if item.startswith('"') and item.endswith('"'):
                    path = item[1:-1]
                elif path is not None and item != "as":
                    alias = item
                    
        if path is None:
            raise ValueError(f"Could not find path in import statement: {items}")
            
        return ImportStmt(path=path, alias=alias)
    
    def func_decl(self, items) -> FuncDecl:
        """func_decl: "func" NAME "(" [param_list] ")" "{" statement* \"}\" """
        # Handle case where we get a FuncDecl back (from statement rule or similar)
        if len(items) == 1 and isinstance(items[0], FuncDecl):
            return items[0]
        
        # Parse the function declaration
        # Expected structure: NAME, [param_list], statement, statement, ...
        if len(items) == 0:
            return FuncDecl(name="unknown", params=[], body=[])
            
        name = str(items[0])
        params = []
        body = []
        
        for i, item in enumerate(items[1:], 1):
            if isinstance(item, list) and all(isinstance(x, str) or (isinstance(x, Token) and x.type == 'NAME') for x in item):
                # This is the parameter list
                params = [str(x) for x in item]
            elif hasattr(item, '__class__') and 'Stmt' in item.__class__.__name__:
                # This is a statement in the function body
                body.append(item)
            
        return self._set_position(FuncDecl(name=name, params=params, body=body), items[0])
    
    def param_list(self, items) -> List[str]:
        """param_list: NAME ("," NAME)*"""
        return [str(item) for item in items]
    
    # =========================================================================
    # L-Values
    # =========================================================================
    
    def lvalue(self, items) -> LValue:
        """lvalue: NAME ("." NAME)?"""
        if len(items) == 1:
            return Identifier(name=str(items[0]))
        else:
            return PropertyAccess(object_name=str(items[0]), property_name=str(items[1]))
    
    # =========================================================================
    # Expressions (following precedence)
    # =========================================================================
    
    def logic(self, items) -> Expr:
        """logic: comparison ( ( "&&" | "||" ) comparison )*"""
        return self._handle_binary_op_chain(items)
    
    def comparison(self, items) -> Expr:
        """comparison: projection ( COMP_OP projection )*"""
        return self._handle_binary_op_chain(items)
    
    def projection(self, items) -> Expr:
        """projection: add ( PROJ_OP type_literal )*"""
        if len(items) == 1:
            return items[0]
        
        # Handle projection: expr | type
        result = items[0]
        i = 1
        while i + 1 < len(items):
            # items[i] should be the "|" operator token
            # items[i + 1] should be the type literal
            target_type = str(items[i + 1])  # The type literal
            result = ProjectionExpr(expr=result, target_type=target_type)
            i += 2
        return result
    
    def add(self, items) -> Expr:
        """add: mul ( ("+"|"-") mul )*"""
        return self._handle_binary_op_chain(items)
    
    def mul(self, items) -> Expr:
        """mul: unary ( ("*"|"/"|"%") unary )*"""
        return self._handle_binary_op_chain(items)
    
    def unary(self, items) -> Expr:
        """unary: ("+"|"-"|"!") unary | primary"""
        if len(items) == 1:
            return items[0]
        elif len(items) == 2:
            op = str(items[0])
            expr = items[1]
            return UnaryOp(op=op, expr=expr)
        else:
            # If more than 2 items, likely nested - take the first meaningful one
            return items[-1]  # Usually the deepest nested result
            
    def _handle_binary_op_chain(self, items) -> Expr:
        """Handle chains of binary operations like a + b - c."""
        if len(items) == 1:
            return items[0]
        elif len(items) == 3:
            # Simple binary operation: left op right
            return BinaryOp(op=str(items[1]), left=items[0], right=items[2])
        else:
            # Chain of operations - left associative
            result = items[0]
            i = 1
            while i + 1 < len(items):
                op = str(items[i])
                right = items[i + 1]
                result = BinaryOp(op=op, left=result, right=right)
                i += 2
            return result
    
    # =========================================================================
    # Primary expressions
    # =========================================================================
    
    def call_expr(self, items) -> CallExpr:
        """call_expr: NAME "(" [arglist] ")" """
        name = str(items[0])
        args = []
        
        # Extract arguments, handling Tree-wrapped expressions
        if len(items) > 1:
            raw_args = items[1] if isinstance(items[1], list) else [items[1]]
            for arg in raw_args:
                args.append(self._unwrap_tree(arg))
        
        return self._set_position(CallExpr(name=name, args=args), items[0])
    
    def method_call(self, items) -> CallExpr:
        """method_call: NAME "." NAME "(" [arglist] ")" """
        module_name = str(items[0])
        method_name = str(items[1])
        args = []
        
        # Extract arguments, handling Tree-wrapped expressions
        if len(items) > 2:
            raw_args = items[2] if isinstance(items[2], list) else [items[2]]
            for arg in raw_args:
                args.append(self._unwrap_tree(arg))
        
        # For now, represent as a qualified function call
        qualified_name = f"{module_name}.{method_name}"
        return self._set_position(CallExpr(name=qualified_name, args=args), items[0])
    
    def arglist(self, items) -> List[Expr]:
        """arglist: expr ("," expr)*"""
        return list(items)
    
    def input_index(self, items) -> InputExpr:
        """input_args: NUMBER -> input_index"""
        index = items[0]
        return InputExpr(index=index, signal_type=None)
    
    def input_typed_index(self, items) -> InputExpr:
        """input_args: type_literal "," NUMBER -> input_typed_index"""
        signal_type = items[0]
        index = items[1]
        return InputExpr(index=index, signal_type=StringLiteral(value=signal_type))
    
    def type_literal(self, items) -> str:
        """type_literal: STRING | NAME"""
        token = items[0]
        if isinstance(token, Token):
            if token.type == 'STRING':
                return token.value[1:-1]  # Remove quotes
            else:  # NAME
                return str(token)
        return str(token)
    
    # =========================================================================
    # Literals and identifiers
    # =========================================================================
    
    def number(self, items) -> NumberLiteral:
        """literal: NUMBER -> number"""
        value = int(items[0])
        return NumberLiteral(value=value)
    
    def string(self, items) -> StringLiteral:
        """literal: STRING -> string"""
        value = str(items[0])[1:-1]  # Remove quotes
        return StringLiteral(value=value)
    
    # =========================================================================
    # Special constructs
    # =========================================================================
    
    def bundle_expr(self, items) -> BundleExpr:
        """bundle_expr: "bundle" "(" [arglist] ")" """
        args = []
        # The arglist should be the first (and only) item when not None
        if len(items) > 0 and items[0] is not None:
            arglist = items[0]
            if isinstance(arglist, list):
                # Unwrap Tree objects to get the actual expressions
                for arg in arglist:
                    args.append(self._unwrap_tree(arg))
            else:
                args = [self._unwrap_tree(arglist)]
        return BundleExpr(exprs=args)
    
    def primary(self, items) -> Expr:
        """Handle primary expressions that weren't caught by other rules."""
        if len(items) == 1:
            item = items[0]
            
            # Handle parenthesized expressions
            if isinstance(item, Expr):
                return item
            
            # Handle identifiers in expression context
            if isinstance(item, Token) and item.type == 'NAME':
                return IdentifierExpr(name=str(item))
            
            # Handle lvalue -> identifier conversion
            if isinstance(item, Identifier):
                return IdentifierExpr(name=item.name)
            
            # Handle property access in expression context  
            if isinstance(item, PropertyAccess):
                # For now, treat as identifier - semantic analysis will handle
                return IdentifierExpr(name=f"{item.object_name}.{item.property_name}")
            
            return item
        
        # Handle specific primary constructs
        if isinstance(items[0], str):
            if items[0] == "input":
                # This should be handled by the input_args rules
                return items[1]  # The InputExpr created by input_args
            elif items[0] == "read":
                memory_name = str(items[1])
                return ReadExpr(memory_name=memory_name)
            elif items[0] == "write":
                memory_name = str(items[1])
                value = items[2]
                return WriteExpr(memory_name=memory_name, value=value)
            elif items[0] == "bundle":
                exprs = items[1] if len(items) > 1 else []
                return BundleExpr(exprs=exprs)
        
        return items[0]


class DSLParser:
    """Main parser class for the Factorio Circuit DSL."""
    
    def __init__(self, grammar_path: Optional[Path] = None):
        """Initialize parser with grammar file."""
        if grammar_path is None:
            # Default to grammar file in project structure
            grammar_path = Path(__file__).parent.parent / "grammar" / "fcdsl.lark"
        
        self.grammar_path = grammar_path
        self.parser = None
        self.transformer = DSLTransformer()
        self._load_grammar()
    
    def _load_grammar(self) -> None:
        """Load and compile the Lark grammar."""
        try:
            with open(self.grammar_path, 'r') as f:
                grammar_text = f.read()
            
            self.parser = Lark(
                grammar_text,
                parser='lalr',
                transformer=self.transformer,
                start='start',
                debug=False
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"Grammar file not found: {self.grammar_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load grammar: {e}")
    
    def parse(self, source_code: str, filename: str = "<string>") -> Program:
        """Parse DSL source code into an AST."""
        try:
            if self.parser is None:
                raise RuntimeError("Parser not initialized")
            
            # Parse and transform in one step
            ast = self.parser.parse(source_code)
            
            if not isinstance(ast, Program):
                raise RuntimeError(f"Expected Program AST node, got {type(ast)}")
            
            return ast
            
        except (ParseError, LexError) as e:
            raise SyntaxError(f"Parse error in {filename}: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error parsing {filename}: {e}")
    
    def parse_file(self, file_path: Path) -> Program:
        """Parse a DSL file into an AST."""
        try:
            with open(file_path, 'r') as f:
                source_code = f.read()
            return self.parse(source_code, str(file_path))
        except FileNotFoundError:
            raise FileNotFoundError(f"Source file not found: {file_path}")


def main():
    """Test the parser with example code."""
    parser = DSLParser()
    
    test_code = """
    let a = input(0);
    let b = input("iron-plate", 1);
    let sum = a + b;
    let result = sum | "signal-output";
    """
    
    try:
        ast = parser.parse(test_code)
        print("Parse successful!")
        print_ast(ast)
    except Exception as e:
        print(f"Parse failed: {e}")


if __name__ == "__main__":
    main()
