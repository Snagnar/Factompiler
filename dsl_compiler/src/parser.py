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

from dsl_compiler.src.dsl_ast import (
    ASTNode, Program, Statement, Expr, LValue,
    DeclStmt, AssignStmt, MemDecl, ExprStmt, ReturnStmt, ImportStmt, FuncDecl,
    Identifier, PropertyAccess,
    BinaryOp, UnaryOp, NumberLiteral, StringLiteral, IdentifierExpr, PropertyAccessExpr,
    CallExpr, InputExpr, ReadExpr, WriteExpr, BundleExpr, ProjectionExpr,
    PlaceExpr, print_ast
)


def preprocess_imports(source_code: str, base_path: Optional[Path] = None) -> str:
    """
    C-style preprocessor that inlines imported files.
    
    Finds import statements of the form:
        import "path/to/file.fcdsl";
    
    And replaces them with the contents of the imported file.
    """
    if base_path is None:
        base_path = Path("tests/sample_programs")
    
    lines = source_code.split('\n')
    processed_lines = []
    processed_files = set()  # Prevent infinite recursion
    
    for line in lines:
        stripped = line.strip()
        
        # Look for import statements (simple regex-free approach)
        if stripped.startswith('import "') and stripped.endswith('";'):
            # Extract the file path
            import_path = stripped[8:-2]  # Remove 'import "' and '";'
            
            # Resolve the file path
            if not import_path.endswith('.fcdsl'):
                import_path += '.fcdsl'
            
            file_path = base_path / import_path
            
            # Avoid circular imports
            if str(file_path) in processed_files:
                processed_lines.append(f"# Skipped circular import: {import_path}")
                continue
                
            try:
                if file_path.exists():
                    processed_files.add(str(file_path))
                    with open(file_path, 'r') as f:
                        imported_content = f.read()
                    
                    # Recursively process imports in the imported file
                    imported_content = preprocess_imports(imported_content, base_path)
                    
                    # Add the imported content with a comment
                    processed_lines.append(f"# --- Imported from {import_path} ---")
                    processed_lines.append(imported_content)
                    processed_lines.append(f"# --- End import {import_path} ---")
                else:
                    # Keep the import statement as-is if file not found (let semantic analysis handle the error)
                    processed_lines.append(line)
            except Exception as e:
                # Keep the import statement if there's an error
                processed_lines.append(line)
        else:
            processed_lines.append(line)
    
    return '\n'.join(processed_lines)


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
    
    def decl_stmt(self, items) -> DeclStmt:
        """decl_stmt: type_name NAME "=" expr"""
        if len(items) == 1 and isinstance(items[0], DeclStmt):
            # Handle nested decl_stmt rule
            return items[0]
        elif len(items) >= 3:
            # items[0] = type_name token, items[1] = NAME token, items[2] = expr
            # The "=" literal is filtered out by Lark
            type_name = str(items[0].value) if hasattr(items[0], 'value') else str(items[0])
            name = str(items[1].value) if hasattr(items[1], 'value') else str(items[1])
            value = self._unwrap_tree(items[2])  # Handle Tree wrapper
            return self._set_position(DeclStmt(type_name=type_name, name=name, value=value), items[1])
        else:
            raise ValueError(f"Unexpected decl_stmt structure: {items}")

    def type_name(self, items) -> str:
        """type_name: INT_KW | SIGNAL_KW | SIGNALTYPE_KW | ENTITY_KW | BUNDLE_TYPE_KW"""
        if len(items) == 1:
            token = items[0]
            return str(token.value) if hasattr(token, 'value') else str(token)
        else:
            raise ValueError(f"Unexpected type_name structure: {items}")
    
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
            
        # With simplified grammar: items[0] = lvalue, items[1] = expr
        # The "=" literal is filtered out by Lark
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
        
        # With simplified grammar: string literals filtered out
        # Remaining items should be: NAME, [expr] (optional expression)
        if len(items) >= 1:
            # First item should be the name
            name = str(items[0])
            init_expr = None
            
            # Look for expression in remaining items
            if len(items) > 1:
                for item in items[1:]:
                    if hasattr(item, '__class__') and ('Expr' in item.__class__.__name__ or 'Literal' in item.__class__.__name__):
                        init_expr = item
                        break
                    elif isinstance(item, Tree):
                        init_expr = self._unwrap_tree(item)
                        break
                        
            return self._set_position(MemDecl(name=name, init_expr=init_expr), items[0])
        else:
            raise ValueError(f"Could not parse mem_decl from items: {items}")
    
    def expr_stmt(self, items) -> ExprStmt:
        """expr_stmt: expr"""
        return ExprStmt(expr=items[0])
    
    def statement_expr_stmt(self, items) -> ExprStmt:
        """Handle expr_stmt at statement level to avoid double nesting."""
        # items[0] should be the ExprStmt from expr_stmt rule
        return items[0]
    
    def return_stmt(self, items) -> ReturnStmt:
        """return_stmt: "return" expr"""
        # Handle case where we get a ReturnStmt back (from statement rule)
        if len(items) == 1 and isinstance(items[0], ReturnStmt):
            return items[0]
        
        # Transform expression if it's still a Tree
        expr = items[0]
        if isinstance(expr, Tree):
            expr = self.transform(expr)
        
        return ReturnStmt(expr=expr)
    
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
            elif hasattr(item, '__class__') and ('Stmt' in item.__class__.__name__ or 
                                                  item.__class__.__name__ in ['MemDecl', 'FuncDecl']):
                # This is a statement in the function body
                body.append(item)
            elif isinstance(item, Tree):
                # Transform any remaining Tree objects
                transformed = self.transform(item)
                if hasattr(transformed, '__class__') and ('Stmt' in transformed.__class__.__name__ or 
                                                          transformed.__class__.__name__ in ['MemDecl', 'FuncDecl']):
                    body.append(transformed)
            
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
    
    def expr(self, items) -> Expr:
        """expr: logic"""
        return items[0]
    
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
        
        # Extract arguments, filtering out None values
        if len(items) > 1 and items[1] is not None:
            raw_args = items[1] if isinstance(items[1], list) else [items[1]]
            for arg in raw_args:
                if arg is not None:
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
        index = NumberLiteral(value=int(items[0]))
        return InputExpr(index=index, signal_type=None)
    
    def input_typed_index(self, items) -> InputExpr:
        """input_args: type_literal "," NUMBER -> input_typed_index"""
        signal_type = items[0]
        index = NumberLiteral(value=int(items[1]))
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
        """bundle_expr: BUNDLE_KW "(" [arglist] ")" """
        args = []
        # With keyword tokens, first item is BUNDLE_KW token, second is arglist (or None)
        start_idx = 1 if len(items) > 0 and hasattr(items[0], 'type') and items[0].type == 'BUNDLE_KW' else 0
        
        if start_idx < len(items) and items[start_idx] is not None:
            arglist = items[start_idx]
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
                return IdentifierExpr(name=item.value)
            
            # Handle lvalue -> identifier conversion
            if isinstance(item, Identifier):
                return IdentifierExpr(name=item.name)
            
            # Handle property access in expression context  
            if isinstance(item, PropertyAccess):
                # Convert to PropertyAccessExpr for expression context
                return PropertyAccessExpr(object_name=item.object_name, property_name=item.property_name)
            
            return item
        
        # Handle keyword token constructs
        if len(items) > 0 and isinstance(items[0], Token):
            token_type = items[0].type
            if token_type == "INPUT_KW":
                return items[1]  # The InputExpr created by input_args rules
            elif token_type == "READ_KW":
                memory_name = str(items[1])
                return ReadExpr(memory_name=memory_name)
            elif token_type == "WRITE_KW":
                memory_name = str(items[1])
                value = items[2]
                return WriteExpr(memory_name=memory_name, value=value)
            elif token_type == "BUNDLE_KW":
                exprs = []
                if len(items) > 1 and items[1] is not None:
                    arglist = items[1]
                    if isinstance(arglist, list):
                        exprs = arglist
                    else:
                        exprs = [arglist]
                return BundleExpr(exprs=exprs)
        
        # Handle string-based constructs (fallback)
        if len(items) > 0 and isinstance(items[0], str):
            if items[0] == "input":
                return items[1]  # The InputExpr created by input_args rules
            elif items[0] == "read":
                memory_name = str(items[1])
                return ReadExpr(memory_name=memory_name)
            elif items[0] == "write":
                memory_name = str(items[1])
                value = items[2]
                return WriteExpr(memory_name=memory_name, value=value)
            elif items[0] == "bundle":
                exprs = []
                if len(items) > 1 and items[1] is not None:
                    arglist = items[1]
                    if isinstance(arglist, list):
                        exprs = arglist
                    else:
                        exprs = [arglist]
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
            
            # Determine base path for imports
            base_path = None
            if filename != "<string>":
                file_path = Path(filename)
                if file_path.is_absolute():
                    base_path = file_path.parent
                else:
                    # Default to sample programs directory
                    base_path = Path("tests/sample_programs")
            else:
                base_path = Path("tests/sample_programs")
            
            # C-style preprocessing: inline imports
            preprocessed_code = preprocess_imports(source_code, base_path)
            
            # Parse and transform in one step
            ast = self.parser.parse(preprocessed_code)
            
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



