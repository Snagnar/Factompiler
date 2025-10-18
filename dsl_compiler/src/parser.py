# parser.py
"""
Parser module for the Factorio Circuit DSL.

Uses Lark parser with the grammar file to parse DSL source code into AST objects.
"""

from pathlib import Path
from typing import List, Optional, Dict, Set

from lark import Lark, Transformer, Tree, Token
from lark.exceptions import ParseError, LexError

from dsl_compiler.src.dsl_ast import (
    ASTNode,
    Program,
    Statement,
    Expr,
    LValue,
    DeclStmt,
    AssignStmt,
    MemDecl,
    ExprStmt,
    ReturnStmt,
    ImportStmt,
    FuncDecl,
    Identifier,
    PropertyAccess,
    BinaryOp,
    UnaryOp,
    NumberLiteral,
    StringLiteral,
    IdentifierExpr,
    PropertyAccessExpr,
    CallExpr,
    ReadExpr,
    WriteExpr,
    ProjectionExpr,
    SignalLiteral,
    DictLiteral,
)


def preprocess_imports(
    source_code: str,
    base_path: Optional[Path] = None,
    processed_files: Optional[Set[Path]] = None,
) -> str:
    """
    C-style preprocessor that inlines imported files.

    Finds import statements of the form:
        import "path/to/file.fcdsl";

    And replaces them with the contents of the imported file.
    """
    if processed_files is None:
        processed_files = set()

    if base_path is None:
        base_path = Path("tests/sample_programs").resolve()
    else:
        base_path = base_path.resolve()

    lines = source_code.split("\n")
    processed_lines = []

    for line in lines:
        stripped = line.strip()

        # Look for import statements (simple regex-free approach)
        if stripped.startswith('import "') and stripped.endswith('";'):
            # Extract the file path
            raw_import_path = stripped[8:-2]  # Remove 'import "' and '";'

            import_path = Path(raw_import_path)
            if import_path.suffix != ".fcdsl":
                import_path = import_path.with_suffix(".fcdsl")

            if not import_path.is_absolute():
                file_path = (base_path / import_path).resolve()
            else:
                file_path = import_path

            # Avoid circular imports
            if file_path in processed_files:
                processed_lines.append(f"# Skipped circular import: {import_path}")
                continue

            try:
                if file_path.exists():
                    processed_files.add(file_path)
                    with open(file_path, "r", encoding="utf-8") as f:
                        imported_content = f.read()

                    # Recursively process imports using the imported file's directory
                    imported_content = preprocess_imports(
                        imported_content,
                        base_path=file_path.parent,
                        processed_files=processed_files,
                    )

                    display_path = (
                        str(import_path)
                        if import_path.is_absolute()
                        else str(import_path)
                    )
                    processed_lines.append(f"# --- Imported from {display_path} ---")
                    processed_lines.append(imported_content)
                    processed_lines.append(f"# --- End import {display_path} ---")
                else:
                    processed_lines.append(line)
            except Exception:
                processed_lines.append(line)
        else:
            processed_lines.append(line)

    return "\n".join(processed_lines)


class DSLTransformer(Transformer):
    """Transforms Lark parse tree into typed AST nodes."""

    def __init__(self):
        super().__init__()
        self.line_info = {}  # Store line/column info

    def _set_position(self, node: ASTNode, token_or_tree) -> ASTNode:
        """Set line/column position on AST node from Lark token/tree."""
        if hasattr(token_or_tree, "meta"):
            node.line = token_or_tree.meta.line
            node.column = token_or_tree.meta.column
        elif hasattr(token_or_tree, "line"):
            node.line = token_or_tree.line
            node.column = token_or_tree.column
        return node

    # =========================================================================
    # Top-level constructs
    # =========================================================================

    def start(self, statements: List[Statement]) -> Program:
        """start: statement*"""
        return Program(statements=statements)

    def decl_stmt(self, items) -> Statement:
        """decl_stmt: type_name NAME "=" expr"""
        if len(items) == 1 and isinstance(items[0], Statement):
            # Handle nested decl_stmt rule (covers DeclStmt and MemDecl)
            return items[0]
        elif len(items) >= 3:
            # items[0] = type_name token, items[1] = NAME token, items[2] = expr
            # The "=" literal is filtered out by Lark
            raw_type = (
                str(items[0].value) if hasattr(items[0], "value") else str(items[0])
            )
            name = str(items[1].value) if hasattr(items[1], "value") else str(items[1])
            value = self._unwrap_tree(items[2])  # Handle Tree wrapper
            type_name = raw_type

            return self._set_position(
                DeclStmt(type_name=type_name, name=name, value=value), items[1]
            )
        else:
            raise ValueError(f"Unexpected decl_stmt structure: {items}")

    def mem_decl(self, items) -> Statement:
        """mem_decl: (Memory|mem) NAME [":" STRING] ["=" expr]"""
        if not items:
            raise ValueError("mem_decl requires a name")

        # Lark may invoke this rule twice (raw tokens, then transformed node).
        if len(items) == 1 and isinstance(items[0], MemDecl):
            return items[0]

        name_token = None
        signal_type: Optional[str] = None
        init_expr: Optional[Expr] = None

        for item in items:
            if isinstance(item, Token):
                if item.type == "NAME":
                    name_token = item
                elif item.type == "STRING":
                    signal_type = item.value[1:-1]
            elif isinstance(item, str):
                if item.startswith("\"") and item.endswith("\""):
                    signal_type = item[1:-1]
            elif isinstance(item, Expr):
                init_expr = item
            else:
                resolved = self._unwrap_tree(item)
                if isinstance(resolved, Expr):
                    init_expr = resolved

        if name_token is None:
            raise ValueError("mem_decl missing memory name token")

        name = str(name_token.value) if hasattr(name_token, "value") else str(name_token)
        mem_node = MemDecl(name=name, signal_type=signal_type, init_expr=init_expr)
        self._set_position(mem_node, name_token)
        return mem_node

    def type_name(self, items) -> str:
        """type_name: INT_KW | SIGNAL_KW | SIGNALTYPE_KW | ENTITY_KW | BUNDLE_TYPE_KW | MEMORY_KW"""
        if len(items) == 1:
            token = items[0]
            value = str(token.value) if hasattr(token, "value") else str(token)
            return value
        else:
            raise ValueError(f"Unexpected type_name structure: {items}")

    def _unwrap_tree(self, item):
        """Unwrap Tree objects and convert tokens to AST nodes."""
        from lark import Tree, Token

        if isinstance(item, Tree):
            if len(item.children) == 1:
                return self._unwrap_tree(item.children[0])
            else:
                return item.children
        elif isinstance(item, Token):
            # Convert tokens to appropriate AST nodes
            if item.type == "STRING":
                # Remove quotes from string literal
                return StringLiteral(
                    value=item.value[1:-1], raw_text=item.value
                )  # Remove surrounding quotes
            elif item.type == "NUMBER":
                return NumberLiteral(value=int(item.value), raw_text=item.value)
            else:
                # For other tokens, return the value
                return item.value
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
                if item.type == "STRING":
                    path = item.value[1:-1]  # Remove quotes
                elif item.type == "NAME" and path is not None:
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
            if isinstance(item, list) and all(
                isinstance(x, str) or (isinstance(x, Token) and x.type == "NAME")
                for x in item
            ):
                # This is the parameter list
                params = [str(x) for x in item]
            elif hasattr(item, "__class__") and (
                "Stmt" in item.__class__.__name__
                or item.__class__.__name__ in ["MemDecl", "FuncDecl"]
            ):
                # This is a statement in the function body
                body.append(item)
            elif isinstance(item, Tree):
                # Transform any remaining Tree objects
                transformed = self.transform(item)
                if hasattr(transformed, "__class__") and (
                    "Stmt" in transformed.__class__.__name__
                    or transformed.__class__.__name__ in ["MemDecl", "FuncDecl"]
                ):
                    body.append(transformed)

        return self._set_position(
            FuncDecl(name=name, params=params, body=body), items[0]
        )

    def param_list(self, items) -> List[str]:
        """param_list: NAME ("," NAME)*"""
        return [str(item) for item in items]

    # =========================================================================
    # L-Values
    # =========================================================================

    def lvalue(self, items) -> LValue:
        """lvalue: NAME ("." NAME)?"""
        if len(items) == 1:
            token = items[0]
            if isinstance(token, Token):
                name = str(token.value)
                raw_text = token.value
            else:
                name = str(token)
                raw_text = name
            return Identifier(name=name, raw_text=raw_text)
        else:
            obj_token, prop_token = items[0], items[1]
            object_name = (
                obj_token.value if isinstance(obj_token, Token) else str(obj_token)
            )
            property_name = (
                prop_token.value if isinstance(prop_token, Token) else str(prop_token)
            )
            raw_text = f"{object_name}.{property_name}"
            return PropertyAccess(
                object_name=object_name,
                property_name=property_name,
                raw_text=raw_text,
            )

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

    def dict_literal(self, items) -> DictLiteral:
        """dict_literal: '{' [dict_item (',' dict_item)*] '}'"""
        entries: Dict[str, Expr] = {}
        first_token = None
        for key, value, token in items:
            entries[key] = value
            if first_token is None:
                first_token = token

        literal = DictLiteral(entries=entries)
        if first_token is not None:
            literal = self._set_position(literal, first_token)
        return literal

    def dict_item(self, items):
        """dict_item: (STRING | NAME) ':' expr"""
        key_token = items[0]
        value_expr = self._unwrap_tree(items[1])

        if isinstance(key_token, Token) and key_token.type == "STRING":
            key = key_token.value[1:-1]
        else:
            key = str(key_token)

        return key, value_expr, key_token

    def signal_with_type(self, items) -> SignalLiteral:
        """signal_literal: "(" type_literal "," expr ")" -> signal_with_type"""
        signal_type = items[0]  # type_literal
        value = self._unwrap_tree(items[1])

        value_raw = getattr(value, "raw_text", None)
        raw_text = f"({signal_type}, {value_raw if value_raw is not None else value})"

        # Unwrap nested signal literals produced by NUMBER grammar so that explicit
        # typed literals carry their numeric payload directly. Without this, we end
        # up lowering a nested, implicitly typed literal which generates an extra
        # virtual signal constant and drops the declared item type.
        if isinstance(value, SignalLiteral) and value.signal_type is None:
            value = value.value

        return SignalLiteral(value=value, signal_type=signal_type, raw_text=raw_text)

    def signal_constant(self, items) -> SignalLiteral:
        """signal_literal: NUMBER -> signal_constant"""
        token = items[0]
        if isinstance(token, Token):
            raw_number = token.value
        else:
            raw_number = str(token)
        value = NumberLiteral(value=int(raw_number), raw_text=raw_number)
        return SignalLiteral(
            value=value, signal_type=None, raw_text=raw_number
        )  # Implicit type

    def type_literal(self, items) -> str:
        """type_literal: STRING | NAME"""
        token = items[0]
        if isinstance(token, Token):
            if token.type == "STRING":
                return token.value[1:-1]  # Remove quotes
            else:  # NAME
                return str(token)
        return str(token)

    # =========================================================================
    # Special constructs
    # =========================================================================

    def primary(self, items) -> Expr:
        """Handle primary expressions that weren't caught by other rules."""
        if len(items) == 1:
            item = items[0]

            # Handle parenthesized expressions
            if isinstance(item, Expr):
                return item

            # Handle identifiers in expression context
            if isinstance(item, Token) and item.type == "NAME":
                return IdentifierExpr(name=item.value, raw_text=item.value)

            # Handle lvalue -> identifier conversion
            if isinstance(item, Identifier):
                return IdentifierExpr(
                    name=item.name,
                    raw_text=getattr(item, "raw_text", item.name),
                )

            # Handle property access in expression context
            if isinstance(item, PropertyAccess):
                # Convert to PropertyAccessExpr for expression context
                return PropertyAccessExpr(
                    object_name=item.object_name,
                    property_name=item.property_name,
                    raw_text=getattr(item, "raw_text", None),
                )

            return item

        # Handle keyword token constructs
        if len(items) > 0 and isinstance(items[0], Token):
            token_type = items[0].type
            if token_type == "READ_KW":
                memory_name = str(items[1])
                return ReadExpr(memory_name=memory_name)
            elif token_type == "WRITE_KW":
                if len(items) >= 3:
                    first_arg = self._unwrap_tree(items[1])
                    second_arg = items[2]

                    # Legacy order: write(memory, value)
                    if isinstance(items[1], Token) and items[1].type == "NAME":
                        memory_name = str(items[1])
                        value_expr = self._unwrap_tree(items[2])
                        write_node = WriteExpr(
                            value=value_expr,
                            memory_name=memory_name,
                            when=None,
                            legacy_syntax=True,
                        )
                        return self._set_position(write_node, items[0])

                    # New order: write(value, memory, when=...)
                    value_expr = first_arg if isinstance(first_arg, Expr) else items[1]

                    if isinstance(second_arg, Token) and second_arg.type == "NAME":
                        memory_name = str(second_arg)
                    elif isinstance(second_arg, Identifier):
                        memory_name = second_arg.name
                    else:
                        memory_name = str(second_arg)

                    when_expr = None
                    when_once = False
                    if len(items) > 3:
                        for extra in items[3:]:
                            if isinstance(extra, Token) and extra.type == "ONCE_KW":
                                when_once = True
                                continue

                            candidate = self._unwrap_tree(extra)

                            if isinstance(candidate, Expr):
                                when_expr = candidate
                                break

                            if isinstance(candidate, str) and candidate == "once":
                                when_once = True

                    if when_once:
                        when_expr = None

                    if not isinstance(value_expr, Expr):
                        value_expr = self._unwrap_tree(value_expr)

                    write_node = WriteExpr(
                        value=value_expr,
                        memory_name=memory_name,
                        when=when_expr,
                        when_once=when_once,
                    )
                    return self._set_position(write_node, items[0])
                error_node = WriteExpr(value=NumberLiteral(0), memory_name="__error")
                return self._set_position(error_node, items[0])

        # Handle string-based constructs (fallback)
        if len(items) > 0 and isinstance(items[0], str):
            if items[0] == "read":
                memory_name = str(items[1])
                return ReadExpr(memory_name=memory_name)
            elif items[0] == "write":
                if len(items) >= 3:
                    first_arg = items[1]
                    second_arg = items[2]

                    # Legacy order fallback
                    if isinstance(first_arg, str) and first_arg.isidentifier():
                        memory_name = first_arg
                        value_expr = self._unwrap_tree(second_arg)
                        return WriteExpr(
                            value=value_expr,
                            memory_name=memory_name,
                            when=None,
                            legacy_syntax=True,
                        )

                    value_expr = self._unwrap_tree(first_arg)
                    memory_name = str(second_arg)
                    when_expr = None
                    when_once = False

                    if len(items) > 3:
                        for extra in items[3:]:
                            candidate = self._unwrap_tree(extra)

                            if isinstance(candidate, Expr):
                                when_expr = candidate
                                break

                            if isinstance(candidate, str) and candidate == "once":
                                when_once = True

                    if when_once:
                        when_expr = None

                    return WriteExpr(
                        value=value_expr,
                        memory_name=memory_name,
                        when=when_expr,
                        when_once=when_once,
                    )

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
            with open(self.grammar_path, "r") as f:
                grammar_text = f.read()

            self.parser = Lark(
                grammar_text,
                parser="lalr",
                transformer=self.transformer,
                start="start",
                debug=False,
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
            if filename != "<string>":
                file_path = Path(filename)
                if not file_path.is_absolute():
                    file_path = (Path.cwd() / file_path).resolve()
                base_path = file_path.parent
            else:
                base_path = Path("tests/sample_programs").resolve()

            # C-style preprocessing: inline imports
            preprocessed_code = preprocess_imports(source_code, base_path)

            # Parse and transform in one step
            ast = self.parser.parse(preprocessed_code)

            # Attach origin filename metadata to all AST nodes for debugging.
            self._attach_source_file(ast, filename)

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
            with open(file_path, "r") as f:
                source_code = f.read()
            return self.parse(source_code, str(file_path))
        except FileNotFoundError:
            raise FileNotFoundError(f"Source file not found: {file_path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _attach_source_file(self, node: ASTNode, filename: str) -> None:
        """Recursively annotate AST nodes with their originating filename."""

        if not isinstance(node, ASTNode):
            return

        if filename:
            node.source_file = filename

        for attr in vars(node).values():
            if isinstance(attr, ASTNode):
                self._attach_source_file(attr, filename)
            elif isinstance(attr, list):
                for item in attr:
                    if isinstance(item, ASTNode):
                        self._attach_source_file(item, filename)
