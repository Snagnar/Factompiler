"""Parse tree transformer producing AST nodes."""

from __future__ import annotations

from lark import Token, Transformer, Tree

from dsl_compiler.src.ast.expressions import (
    BinaryOp,
    BundleAllExpr,
    BundleAnyExpr,
    BundleLiteral,
    BundleSelectExpr,
    CallExpr,
    EntityOutputExpr,
    IdentifierExpr,
    ProjectionExpr,
    PropertyAccessExpr,
    ReadExpr,
    SignalLiteral,
    SignalTypeAccess,
    UnaryOp,
    WriteExpr,
)
from dsl_compiler.src.ast.literals import (
    DictLiteral,
    Identifier,
    NumberLiteral,
    PropertyAccess,
    StringLiteral,
)
from dsl_compiler.src.ast.statements import (
    AssignStmt,
    ASTNode,
    DeclStmt,
    Expr,
    ExprStmt,
    ForStmt,
    FuncDecl,
    ImportStmt,
    LValue,
    MemDecl,
    Program,
    ReturnStmt,
    Statement,
    TypedParam,
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

    def start(self, statements: list[Statement]) -> Program:
        """start: statement*"""
        return Program(statements=statements)

    def decl_stmt(self, items) -> Statement:
        """decl_stmt: type_name NAME "=" expr"""
        if len(items) == 1 and isinstance(items[0], Statement):
            return items[0]
        if len(items) >= 3:
            raw_type = str(items[0].value) if hasattr(items[0], "value") else str(items[0])
            name = str(items[1].value) if hasattr(items[1], "value") else str(items[1])
            value = self._unwrap_tree(items[2])
            type_name = raw_type

            result = self._set_position(
                DeclStmt(type_name=type_name, name=name, value=value), items[1]
            )
            assert isinstance(result, Statement)
            return result
        raise ValueError(f"Unexpected decl_stmt structure: {items}")

    def mem_decl(self, items) -> Statement:
        """mem_decl: MEMORY_KW NAME [":" STRING]"""
        if not items:
            raise ValueError("mem_decl requires a name")

        if len(items) == 1 and isinstance(items[0], MemDecl):
            return items[0]

        name_token = None
        signal_type: str | None = None

        for item in items:
            if isinstance(item, Token):
                if item.type == "NAME":
                    name_token = item
                elif item.type == "STRING":
                    signal_type = item.value[1:-1]
            elif isinstance(item, str) and item.startswith('"') and item.endswith('"'):
                signal_type = item[1:-1]

        if name_token is None:
            raise ValueError("mem_decl missing memory name token")

        name = str(name_token.value) if hasattr(name_token, "value") else str(name_token)
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
                result = NumberLiteral(value=self._parse_number(item.value), raw_text=item.value)
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
        """func_decl: "func" NAME "(" [typed_param_list] ")" "{" statement* "}" """
        if len(items) == 1 and isinstance(items[0], FuncDecl):
            return items[0]

        if len(items) == 0:
            return FuncDecl(name="unknown", params=[], body=[])

        name = str(items[0])
        params: list[TypedParam] = []
        body: list[Statement] = []

        for item in items[1:]:
            if isinstance(item, list) and all(isinstance(x, TypedParam) for x in item):
                params = item
            elif isinstance(item, TypedParam):
                params = [item]  # Single parameter case
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

        result = self._set_position(FuncDecl(name=name, params=params, body=body), items[0])
        assert isinstance(result, FuncDecl)
        return result

    def typed_param_list(self, items) -> list[TypedParam]:
        """typed_param_list: typed_param ("," typed_param)*"""
        return list(items)

    def typed_param(self, items) -> TypedParam:
        """typed_param: param_type NAME"""
        type_token = items[0]
        name_token = items[1]

        type_name = str(type_token.value) if hasattr(type_token, "value") else str(type_token)
        param_name = str(name_token.value) if hasattr(name_token, "value") else str(name_token)

        result = TypedParam(type_name=type_name, name=param_name)
        # Set position directly since TypedParam doesn't inherit from ASTNode
        if hasattr(name_token, "line"):
            result.line = name_token.line
            result.column = name_token.column
        return result

    def for_stmt(self, items) -> ForStmt:
        """for_stmt: FOR_KW NAME IN_KW for_iterator "{" statement* "}"""
        if len(items) == 1 and isinstance(items[0], ForStmt):
            return items[0]

        # Find iterator name (first NAME token after FOR_KW)
        iterator_name = None
        iterator_data = None
        statements = []
        position_token = None

        for item in items:
            if isinstance(item, Token):
                if item.type == "NAME" and iterator_name is None:
                    iterator_name = str(item.value)
                    position_token = item
            elif isinstance(item, dict):
                # This is the iterator data from range_iterator or list_iterator
                iterator_data = item
            elif isinstance(item, Statement):
                statements.append(item)
            elif isinstance(item, Tree) and item.data in (
                "range_iterator",
                "list_iterator",
                "for_iterator",
            ):
                # Could be for_iterator tree
                iterator_data = self.transform(item)

        if iterator_name is None:
            raise ValueError(f"Could not find iterator name in for_stmt: {items}")
        if iterator_data is None:
            raise ValueError(f"Could not find iterator data in for_stmt: {items}")

        # Unwrap for_iterator if needed
        if isinstance(iterator_data, dict) and "inner" in iterator_data:
            iterator_data = iterator_data["inner"]

        # Build ForStmt based on iterator type
        if "values" in iterator_data:
            # List iterator
            result = self._set_position(
                ForStmt(
                    iterator_name=iterator_name,
                    start=None,
                    stop=None,
                    step=None,
                    values=iterator_data["values"],
                    body=statements,
                ),
                position_token,
            )
            assert isinstance(result, ForStmt)
            return result
        else:
            # Range iterator
            result = self._set_position(
                ForStmt(
                    iterator_name=iterator_name,
                    start=iterator_data["start"],
                    stop=iterator_data["stop"],
                    step=iterator_data.get("step", 1),
                    values=None,
                    body=statements,
                ),
                position_token,
            )
            assert isinstance(result, ForStmt)
            return result

    def for_iterator(self, items):
        """for_iterator: range_iterator | list_iterator"""
        if len(items) == 1:
            return items[0]
        return items[0]

    def range_bound(self, items):
        """range_bound: NUMBER | NAME

        Returns either an int (for NUMBER) or a string (for NAME variable reference).
        """
        item = items[0]
        if isinstance(item, Token):
            if item.type == "NUMBER":
                return self._parse_number(item.value)
            elif item.type == "NAME":
                return str(item.value)
        # Fallback for already-transformed items
        if isinstance(item, int):
            return item
        if isinstance(item, str):
            return item
        raise ValueError(f"Unexpected range_bound item: {item}")

    def range_iterator(self, items) -> dict:
        """range_iterator: range_bound RANGE_OP range_bound [STEP_KW range_bound]"""
        # Extract bounds - can be int or str (variable name)
        bounds = []
        step_value: int | str = 1

        i = 0
        while i < len(items):
            item = items[i]
            if isinstance(item, Token) and item.type == "STEP_KW":
                # Next item should be the step value
                if i + 1 < len(items):
                    step_value = items[i + 1]
                    i += 1
            elif isinstance(item, Token):
                # Skip other tokens (like RANGE_OP) - Token inherits from str so must check first
                pass
            elif isinstance(item, (int, str)):
                bounds.append(item)
            i += 1

        if len(bounds) < 2:
            raise ValueError(f"Range iterator requires start and stop: {items}")

        return {"start": bounds[0], "stop": bounds[1], "step": step_value}

    def list_iterator(self, items) -> dict:
        """list_iterator: "[" [NUMBER ("," NUMBER)*] "]" """
        values = []
        for item in items:
            if isinstance(item, Token) and item.type == "NUMBER":
                values.append(self._parse_number(item.value))
        return {"values": values}

    def param_type(self, items) -> str:
        """param_type: INT_KW | SIGNAL_KW | ENTITY_KW"""
        token = items[0]
        return str(token.value) if hasattr(token, "value") else str(token)

    def lvalue(self, items) -> LValue:
        """lvalue: NAME ("." NAME)?"""
        result: LValue
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
        object_name = obj_token.value if isinstance(obj_token, Token) else str(obj_token)
        property_name = prop_token.value if isinstance(prop_token, Token) else str(prop_token)
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
        return self._handle_binary_op_chain(items, normalize_op=self._normalize_logical_op)

    def logic_and(self, items) -> Expr:
        """logic_and: output_spec ( LOGIC_AND output_spec )*"""
        return self._handle_binary_op_chain(items, normalize_op=self._normalize_logical_op)

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
        result = self._set_position(node, condition)
        assert isinstance(result, Expr)
        return result

    def output_value(self, items) -> Expr:
        """output_value: primary"""
        result = items[0]
        # Unwrap any Tree/Token objects to get proper AST nodes
        if not isinstance(result, Expr):
            result = self._unwrap_tree(result)
        assert isinstance(result, Expr)
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
            target_type = items[index + 1]
            # Keep SignalTypeAccess as-is; convert other types to string
            if not isinstance(target_type, SignalTypeAccess):
                target_type = str(target_type)
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

        result = self._set_position(CallExpr(name=name, args=args), items[0])
        assert isinstance(result, CallExpr)
        return result

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
        result = self._set_position(CallExpr(name=qualified_name, args=args), items[0])
        assert isinstance(result, CallExpr)
        return result

    def memory_read(self, items) -> ReadExpr:
        """memory_read: NAME "." "read" "(" ")" """
        memory_name = str(items[0])
        result = self._set_position(ReadExpr(memory_name=memory_name), items[0])
        assert isinstance(result, ReadExpr)
        return result

    def memory_write(self, items) -> WriteExpr:
        """memory_write: NAME "." "write" "(" expr ")" """
        memory_name = str(items[0])
        value = self._unwrap_tree(items[1])
        write_node = WriteExpr(value=value, memory_name=memory_name, when=None)
        result = self._set_position(write_node, items[0])
        assert isinstance(result, WriteExpr)
        return result

    def memory_write_when(self, items) -> WriteExpr:
        """memory_write_when: NAME "." "write" "(" expr "," WHEN_KW "=" expr ")"

        Items: [NAME, expr, WHEN_KW, expr]
        """
        memory_name = str(items[0])
        value = self._unwrap_tree(items[1])
        # items[2] is the WHEN_KW token, items[3] is the condition expression
        condition = self._unwrap_tree(items[3])
        write_node = WriteExpr(value=value, memory_name=memory_name, when=condition)
        result = self._set_position(write_node, items[0])
        assert isinstance(result, WriteExpr)
        return result

    def memory_latch_write(self, items) -> WriteExpr:
        """memory_latch_write: NAME "." "write" "(" expr "," latch_kwargs ")"

        Items: [NAME, expr, (set_expr, reset_expr, set_priority)]
        """
        memory_name = str(items[0])
        value = self._unwrap_tree(items[1])
        set_expr, reset_expr, set_priority = items[2]
        write_node = WriteExpr(
            value=value,
            memory_name=memory_name,
            set_signal=set_expr,
            reset_signal=reset_expr,
            set_priority=set_priority,
        )
        result = self._set_position(write_node, items[0])
        assert isinstance(result, WriteExpr)
        return result

    def latch_set_reset(self, items) -> tuple:
        """latch_kwargs: SET_KW "=" expr "," RESET_KW "=" expr -> latch_set_reset

        SR latch (set priority) - set comes first.
        Items: [SET_KW, expr, RESET_KW, expr]
        """
        set_expr = self._unwrap_tree(items[1])
        reset_expr = self._unwrap_tree(items[3])
        return (set_expr, reset_expr, True)  # set_priority=True (SR latch)

    def latch_reset_set(self, items) -> tuple:
        """latch_kwargs: RESET_KW "=" expr "," SET_KW "=" expr -> latch_reset_set

        RS latch (reset priority) - reset comes first.
        Items: [RESET_KW, expr, SET_KW, expr]
        """
        reset_expr = self._unwrap_tree(items[1])
        set_expr = self._unwrap_tree(items[3])
        return (set_expr, reset_expr, False)  # set_priority=False (RS latch)

    def arglist(self, items) -> list[Expr]:
        """arglist: expr ("," expr)*"""
        return list(items)

    def dict_literal(self, items) -> DictLiteral:
        """dict_literal: '{' [dict_item (',' dict_item)*] '}'"""
        entries: dict[str, Expr] = {}
        first_token = None
        for key, value, token in items:
            entries[key] = value
            if first_token is None:
                first_token = token

        literal: DictLiteral = DictLiteral(entries=entries)
        if first_token is not None:
            result = self._set_position(literal, first_token)
            assert isinstance(result, DictLiteral)
            return result
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

    def empty_brace(self, items) -> BundleLiteral:
        """empty_brace: '{' '}'

        Empty braces default to an empty bundle (since empty dicts make no sense).
        """
        return BundleLiteral(elements=[])

    def bundle_literal(self, items) -> BundleLiteral:
        """bundle_literal: '{' [bundle_element (',' bundle_element)*] '}'"""
        elements: list[Expr] = []
        first_token = None
        for item in items:
            unwrapped = self._unwrap_tree(item)
            if unwrapped is not None:
                elements.append(unwrapped)
            if first_token is None and hasattr(item, "line"):
                first_token = item

        literal: BundleLiteral = BundleLiteral(elements=elements)
        if first_token is not None:
            result = self._set_position(literal, first_token)
            assert isinstance(result, BundleLiteral)
            return result
        return literal

    def bundle_element(self, items) -> Expr:
        """bundle_element: expr"""
        return self._unwrap_tree(items[0])

    def bundle_select_chain(self, items) -> Expr:
        """bundle_select_chain: atom ( '[' STRING ']' )*

        Handles chained subscript operations like: entity.output["signal-A"]
        items[0] is the base expression (atom)
        items[1:] are STRING tokens for each subscript operation
        """
        base = self._unwrap_tree(items[0])

        # Convert LValue types to Expr types
        if isinstance(base, Identifier):
            current_expr: Expr = IdentifierExpr(
                name=base.name,
                raw_text=getattr(base, "raw_text", base.name),
            )
        elif isinstance(base, PropertyAccess):
            # Check for entity.output - creates EntityOutputExpr
            if base.property_name == "output":
                current_expr = EntityOutputExpr(
                    entity_name=base.object_name,
                    raw_text=getattr(base, "raw_text", None),
                )
            else:
                current_expr = PropertyAccessExpr(
                    object_name=base.object_name,
                    property_name=base.property_name,
                    raw_text=getattr(base, "raw_text", None),
                )
        elif isinstance(base, Expr):
            current_expr = base
        else:
            # Fallback - shouldn't happen but handle gracefully
            current_expr = base  # type: ignore

        # Apply each subscript operation left-to-right
        for signal_type_token in items[1:]:
            if isinstance(signal_type_token, Token):
                signal_type = signal_type_token.value[1:-1]  # Remove quotes
            else:
                signal_type = str(signal_type_token)[1:-1]

            select_expr = BundleSelectExpr(bundle=current_expr, signal_type=signal_type)
            # Set position from the signal type token if available
            if hasattr(signal_type_token, "line"):
                self._set_position(select_expr, signal_type_token)
            current_expr = select_expr

        return current_expr

    def bundle_any(self, items) -> BundleAnyExpr:
        """bundle_any: ANY_KW '(' expr ')'

        items[0] is the ANY_KW token, items[1] is the bundle expression.
        """
        bundle_expr = self._unwrap_tree(items[1])
        any_expr: BundleAnyExpr = BundleAnyExpr(bundle=bundle_expr)
        if hasattr(items[0], "line"):
            result = self._set_position(any_expr, items[0])
            assert isinstance(result, BundleAnyExpr)
            return result
        return any_expr

    def bundle_all(self, items) -> BundleAllExpr:
        """bundle_all: ALL_KW '(' expr ')'

        items[0] is the ALL_KW token, items[1] is the bundle expression.
        """
        bundle_expr = self._unwrap_tree(items[1])
        all_expr: BundleAllExpr = BundleAllExpr(bundle=bundle_expr)
        if hasattr(items[0], "line"):
            result = self._set_position(all_expr, items[0])
            assert isinstance(result, BundleAllExpr)
            return result
        return all_expr

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

    def type_property_access(self, items) -> SignalTypeAccess:
        """type_property_access: NAME "." NAME

        Returns a SignalTypeAccess node that will be resolved during semantic analysis
        to extract the signal type from a signal variable.
        """
        object_name = str(items[0])
        property_name = str(items[1])
        return SignalTypeAccess(object_name=object_name, property_name=property_name)

    def type_literal(self, items) -> str | SignalTypeAccess:
        """type_literal: STRING | type_property_access | NAME"""
        token = items[0]
        # If it's a SignalTypeAccess from type_property_access, return it as-is
        if isinstance(token, SignalTypeAccess):
            return token
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
                # Check for entity.output - creates EntityOutputExpr
                if item.property_name == "output":
                    return EntityOutputExpr(
                        entity_name=item.object_name,
                        raw_text=getattr(item, "raw_text", None),
                    )
                return PropertyAccessExpr(
                    object_name=item.object_name,
                    property_name=item.property_name,
                    raw_text=getattr(item, "raw_text", None),
                )

            return item

        return items[0]
