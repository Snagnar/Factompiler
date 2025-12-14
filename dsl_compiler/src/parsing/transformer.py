"""Parse tree transformer producing AST nodes."""

from __future__ import annotations

from typing import Dict, List, Optional

from lark import Transformer, Tree, Token

from dsl_compiler.src.ast.statements import (
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
)
from dsl_compiler.src.ast.literals import (
    Identifier,
    PropertyAccess,
    NumberLiteral,
    StringLiteral,
    DictLiteral,
)
from dsl_compiler.src.ast.expressions import (
    BinaryOp,
    UnaryOp,
    IdentifierExpr,
    PropertyAccessExpr,
    CallExpr,
    ReadExpr,
    WriteExpr,
    ProjectionExpr,
    SignalLiteral,
)


class DSLTransformer(Transformer):
    """Transforms Lark parse tree into typed AST nodes."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def _parse_number(text: str) -> int:
        """Parse a number literal supporting binary, octal, hex, and decimal formats.

        Args:
            text: The number string (e.g., "0xFF", "0b1010", "0o755", "42", "-10")

        Returns:
            The integer value
        """
        text = text.strip()

        # Handle different bases
        if text.startswith(("0x", "0X")):
            return int(text, 16)
        elif text.startswith(("0o", "0O")):
            return int(text, 8)
        elif text.startswith(("0b", "0B")):
            return int(text, 2)
        else:
            return int(text, 10)

    def _set_position(self, node: ASTNode, token_or_tree) -> ASTNode:
        """Set line/column position on AST node from Lark token/tree."""
        if hasattr(token_or_tree, "meta"):
            node.line = token_or_tree.meta.line
            node.column = token_or_tree.meta.column
        elif hasattr(token_or_tree, "line"):
            node.line = token_or_tree.line
            node.column = token_or_tree.column
        return node

    def start(self, statements: List[Statement]) -> Program:
        """start: statement*"""
        return Program(statements=statements)

    def decl_stmt(self, items) -> Statement:
        """decl_stmt: type_name NAME "=" expr"""
        if len(items) == 1 and isinstance(items[0], Statement):
            return items[0]
        if len(items) >= 3:
            raw_type = (
                str(items[0].value) if hasattr(items[0], "value") else str(items[0])
            )
            name = str(items[1].value) if hasattr(items[1], "value") else str(items[1])
            value = self._unwrap_tree(items[2])
            type_name = raw_type

            return self._set_position(
                DeclStmt(type_name=type_name, name=name, value=value), items[1]
            )
        raise ValueError(f"Unexpected decl_stmt structure: {items}")

    def mem_decl(self, items) -> Statement:
        """mem_decl: (Memory|mem) NAME [":" STRING]"""
        if not items:
            raise ValueError("mem_decl requires a name")

        if len(items) == 1 and isinstance(items[0], MemDecl):
            return items[0]

        name_token = None
        signal_type: Optional[str] = None

        for item in items:
            if isinstance(item, Token):
                if item.type == "NAME":
                    name_token = item
                elif item.type == "STRING":
                    signal_type = item.value[1:-1]
            elif isinstance(item, str):
                if item.startswith('"') and item.endswith('"'):
                    signal_type = item[1:-1]

        if name_token is None:
            raise ValueError("mem_decl missing memory name token")

        name = (
            str(name_token.value) if hasattr(name_token, "value") else str(name_token)
        )
        mem_node = MemDecl(name=name, signal_type=signal_type)
        self._set_position(mem_node, name_token)
        return mem_node

    def type_name(self, items) -> str:
        """type_name: INT_KW | SIGNAL_KW | SIGNALTYPE_KW | ENTITY_KW | BUNDLE_TYPE_KW | MEMORY_KW"""
        if len(items) == 1:
            token = items[0]
            value = str(token.value) if hasattr(token, "value") else str(token)
            return value
        raise ValueError(f"Unexpected type_name structure: {items}")

    def _unwrap_tree(self, item):
        """Unwrap Tree objects and convert tokens to AST nodes."""
        if isinstance(item, Tree):
            if len(item.children) == 1:
                return self._unwrap_tree(item.children[0])
            return item.children
        if isinstance(item, Token):
            if item.type == "STRING":
                result = StringLiteral(value=item.value[1:-1], raw_text=item.value)
                return self._set_position(result, item)
            if item.type == "NUMBER":
                result = NumberLiteral(
                    value=self._parse_number(item.value), raw_text=item.value
                )
                return self._set_position(result, item)
            return item.value
        return item

    def assign_stmt(self, items) -> AssignStmt:
        """assign_stmt: lvalue "=" expr"""
        if len(items) == 1 and isinstance(items[0], AssignStmt):
            stmt = items[0]
            stmt.value = self._unwrap_tree(stmt.value)
            return stmt

        if len(items) >= 2:
            target = items[0]
            value = self._unwrap_tree(items[1])
            stmt = AssignStmt(target=target, value=value)
            # Set position from target (first element)
            if hasattr(target, "line") and target.line > 0:
                stmt.line = target.line
                stmt.column = getattr(target, "column", 0)
                stmt.source_file = getattr(target, "source_file", None)
            return stmt
        raise ValueError(f"Could not parse assign_stmt from items: {items}")

    def expr_stmt(self, items) -> ExprStmt:
        """expr_stmt: expr"""
        return ExprStmt(expr=items[0])

    def statement_expr_stmt(self, items) -> ExprStmt:
        """Handle expr_stmt at statement level to avoid double nesting."""
        return items[0]

    def return_stmt(self, items) -> ReturnStmt:
        """return_stmt: "return" expr"""
        if len(items) == 1 and isinstance(items[0], ReturnStmt):
            return items[0]

        expr = items[0]
        if isinstance(expr, Tree):
            expr = self.transform(expr)

        return ReturnStmt(expr=expr)

    def import_stmt(self, items) -> ImportStmt:
        """import_stmt: "import" STRING ["as" NAME]"""
        if len(items) == 1 and isinstance(items[0], ImportStmt):
            return items[0]

        path = None
        alias = None

        for item in items:
            if isinstance(item, Token):
                if item.type == "STRING":
                    path = item.value[1:-1]
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
        """func_decl: "func" NAME "(" [param_list] ")" "{" statement* "}" """
        if len(items) == 1 and isinstance(items[0], FuncDecl):
            return items[0]

        if len(items) == 0:
            return FuncDecl(name="unknown", params=[], body=[])

        name = str(items[0])
        params = []
        body = []

        for item in items[1:]:
            if isinstance(item, list) and all(
                isinstance(x, str) or (isinstance(x, Token) and x.type == "NAME")
                for x in item
            ):
                params = [str(x) for x in item]
            elif hasattr(item, "__class__") and (
                "Stmt" in item.__class__.__name__
                or item.__class__.__name__ in ["MemDecl", "FuncDecl"]
            ):
                body.append(item)
            elif isinstance(item, Tree):
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
            result = Identifier(name=name, raw_text=raw_text)
            self._set_position(result, token)
            return result

        obj_token, prop_token = items[0], items[1]
        object_name = (
            obj_token.value if isinstance(obj_token, Token) else str(obj_token)
        )
        property_name = (
            prop_token.value if isinstance(prop_token, Token) else str(prop_token)
        )
        raw_text = f"{object_name}.{property_name}"
        result = PropertyAccess(
            object_name=object_name,
            property_name=property_name,
            raw_text=raw_text,
        )
        self._set_position(result, obj_token)
        return result

    def expr(self, items) -> Expr:
        """expr: logic_or"""
        return items[0]

    def logic_or(self, items) -> Expr:
        """logic_or: logic_and ( LOGIC_OR logic_and )*"""
        return self._handle_binary_op_chain(
            items, normalize_op=self._normalize_logical_op
        )

    def logic_and(self, items) -> Expr:
        """logic_and: output_spec ( LOGIC_AND output_spec )*"""
        return self._handle_binary_op_chain(
            items, normalize_op=self._normalize_logical_op
        )

    def output_spec(self, items) -> Expr:
        """output_spec: comparison [ OUTPUT_SPEC output_value ]"""
        # Filter out None items (optional parts not present)
        items = [item for item in items if item is not None]

        if len(items) == 1:
            # No output specifier - just return the comparison
            return items[0]

        # Has output specifier
        # items = [comparison, ":", output_value]
        from dsl_compiler.src.ast.expressions import OutputSpecExpr

        condition = items[0]
        # Skip the ":" token (items[1]) and get the output_value (items[2])
        output_value = items[2] if len(items) > 2 else items[1]

        node = OutputSpecExpr(condition=condition, output_value=output_value)
        return self._set_position(node, condition)

    def output_value(self, items) -> Expr:
        """output_value: primary"""
        result = items[0]
        # Unwrap any Tree/Token objects to get proper AST nodes
        if not isinstance(result, Expr):
            result = self._unwrap_tree(result)
        return result

    def comparison(self, items) -> Expr:
        """comparison: projection ( COMP_OP projection )*"""
        return self._handle_binary_op_chain(items)

    def projection(self, items) -> Expr:
        """projection: bitwise_or ( PROJ_OP type_literal )*"""
        if len(items) == 1:
            return items[0]

        result = items[0]
        index = 1
        while index + 1 < len(items):
            target_type = str(items[index + 1])
            result = ProjectionExpr(expr=result, target_type=target_type)
            index += 2
        return result

    def bitwise_or(self, items) -> Expr:
        """bitwise_or: bitwise_xor ( BITWISE_OR bitwise_xor )*"""
        return self._handle_binary_op_chain(items)

    def bitwise_xor(self, items) -> Expr:
        """bitwise_xor: bitwise_and ( BITWISE_XOR bitwise_and )*"""
        return self._handle_binary_op_chain(items)

    def bitwise_and(self, items) -> Expr:
        """bitwise_and: shift ( BITWISE_AND shift )*"""
        return self._handle_binary_op_chain(items)

    def shift(self, items) -> Expr:
        """shift: add ( SHIFT_OP add )*"""
        return self._handle_binary_op_chain(items)

    def add(self, items) -> Expr:
        """add: mul ( ADD_OP mul )*"""
        return self._handle_binary_op_chain(items)

    def mul(self, items) -> Expr:
        """mul: power ( MUL_OP power )*"""
        return self._handle_binary_op_chain(items)

    def power(self, items) -> Expr:
        """power: unary ( POWER_OP power )?

        Note: Right-associative due to recursive grammar structure.
        """
        if len(items) == 1:
            return items[0]

        # items = [unary, "**", power] - operator is in the middle
        left = items[0]
        # Skip the operator token (items[1])
        right = items[2] if len(items) > 2 else items[1]

        node = BinaryOp(op="**", left=left, right=right)

        if hasattr(left, "line") and left.line > 0:
            node.line = left.line
            node.column = getattr(left, "column", 0)

        return node

    def unary(self, items) -> Expr:
        """unary: ("+"|"-"|"!") unary | primary"""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            op = str(items[0])
            expr = items[1]
            return UnaryOp(op=op, expr=expr)
        return items[-1]

    def _normalize_logical_op(self, op: str) -> str:
        """Normalize logical operator keywords to symbols."""
        if op.lower() == "and":
            return "&&"
        if op.lower() == "or":
            return "||"
        return op

    def _handle_binary_op_chain(self, items, normalize_op=None) -> Expr:
        """Handle chains of binary operations like a + b - c.

        Args:
            items: List of operands and operators
            normalize_op: Optional function to normalize operator strings
        """
        if len(items) == 1:
            return items[0]
        if len(items) == 3:
            op = str(items[1])
            if normalize_op:
                op = normalize_op(op)
            node = BinaryOp(op=op, left=items[0], right=items[2])
            # Set position from first operand if it has position info
            if hasattr(items[0], "line") and items[0].line > 0:
                node.line = items[0].line
                node.column = getattr(items[0], "column", 0)
                node.source_file = getattr(items[0], "source_file", None)
            return node

        result = items[0]
        index = 1
        while index + 1 < len(items):
            op = str(items[index])
            if normalize_op:
                op = normalize_op(op)
            right = items[index + 1]
            node = BinaryOp(op=op, left=result, right=right)
            # Set position from left operand (first in chain)
            if hasattr(result, "line") and result.line > 0:
                node.line = result.line
                node.column = getattr(result, "column", 0)
                node.source_file = getattr(result, "source_file", None)
            result = node
            index += 2
        return result

    def call_expr(self, items) -> CallExpr:
        """call_expr: NAME "(" [arglist] ")" """
        name = str(items[0])
        args = []

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

        if len(items) > 2:
            raw_args = items[2] if isinstance(items[2], list) else [items[2]]
            for arg in raw_args:
                args.append(self._unwrap_tree(arg))

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
        signal_type = items[0]
        value = self._unwrap_tree(items[1])

        value_raw = getattr(value, "raw_text", None)
        raw_text = f"({signal_type}, {value_raw if value_raw is not None else value})"

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
        value = NumberLiteral(value=self._parse_number(raw_number), raw_text=raw_number)
        return SignalLiteral(value=value, signal_type=None, raw_text=raw_number)

    def type_literal(self, items) -> str:
        """type_literal: STRING | NAME"""
        token = items[0]
        if isinstance(token, Token):
            if token.type == "STRING":
                return token.value[1:-1]
            return str(token)
        return str(token)

    def primary(self, items) -> Expr:
        """Handle primary expressions that weren't caught by other rules."""
        if len(items) == 1:
            item = items[0]

            if isinstance(item, Expr):
                return item

            if isinstance(item, Token) and item.type == "NAME":
                return IdentifierExpr(name=item.value, raw_text=item.value)

            if isinstance(item, Identifier):
                return IdentifierExpr(
                    name=item.name,
                    raw_text=getattr(item, "raw_text", item.name),
                )

            if isinstance(item, PropertyAccess):
                return PropertyAccessExpr(
                    object_name=item.object_name,
                    property_name=item.property_name,
                    raw_text=getattr(item, "raw_text", None),
                )

            return item

        if len(items) > 0 and isinstance(items[0], Token):
            token_type = items[0].type
            if token_type == "READ_KW":
                memory_name = str(items[1])
                return ReadExpr(memory_name=memory_name)
            if token_type == "WRITE_KW":
                if len(items) >= 3:
                    first_arg = self._unwrap_tree(items[1])
                    second_arg = items[2]

                    value_expr = first_arg if isinstance(first_arg, Expr) else items[1]

                    if isinstance(second_arg, Token) and second_arg.type == "NAME":
                        memory_name = str(second_arg)
                    elif isinstance(second_arg, Identifier):
                        memory_name = second_arg.name
                    else:
                        memory_name = str(second_arg)

                    when_expr = None
                    if len(items) > 3:
                        for extra in items[3:]:
                            candidate = self._unwrap_tree(extra)

                            if isinstance(candidate, Expr):
                                when_expr = candidate
                                break

                    if not isinstance(value_expr, Expr):
                        value_expr = self._unwrap_tree(value_expr)

                    write_node = WriteExpr(
                        value=value_expr,
                        memory_name=memory_name,
                        when=when_expr,
                    )
                    return self._set_position(write_node, items[0])
                error_node = WriteExpr(value=NumberLiteral(0), memory_name="__error")
                return self._set_position(error_node, items[0])

        if len(items) > 0 and isinstance(items[0], str):
            if items[0] == "read":
                memory_name = str(items[1])
                return ReadExpr(memory_name=memory_name)
            if items[0] == "write":
                if len(items) >= 3:
                    first_arg = items[1]
                    second_arg = items[2]

                    value_expr = self._unwrap_tree(first_arg)
                    memory_name = str(second_arg)
                    when_expr = None

                    if len(items) > 3:
                        for extra in items[3:]:
                            candidate = self._unwrap_tree(extra)

                            if isinstance(candidate, Expr):
                                when_expr = candidate
                                break

                    return WriteExpr(
                        value=value_expr,
                        memory_name=memory_name,
                        when=when_expr,
                    )

        return items[0]
