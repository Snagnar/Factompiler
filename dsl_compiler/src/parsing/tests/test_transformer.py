"""
Tests for parsing/transformer.py - Lark tree to AST transformation.
"""

import pytest
from lark import Token, Tree

from dsl_compiler.src.ast.expressions import (
    BinaryOp,
    BundleAllExpr,
    BundleAnyExpr,
    BundleLiteral,
    BundleSelectExpr,
    CallExpr,
    EntityOutputExpr,
    IdentifierExpr,
    OutputSpecExpr,
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
)
from dsl_compiler.src.ast.statements import (
    AssignStmt,
    DeclStmt,
    ExprStmt,
    ForStmt,
    FuncDecl,
    ImportStmt,
    MemDecl,
    Program,
    ReturnStmt,
    TypedParam,
)
from dsl_compiler.src.parsing.transformer import DSLTransformer


class TestDSLTransformerInit:
    """Tests for DSLTransformer.__init__"""

    def test_init(self):
        """Test DSLTransformer initialization."""
        transformer = DSLTransformer()
        assert transformer is not None
        assert isinstance(transformer, DSLTransformer)


class TestDSLTransformerParseNumber:
    """Tests for DSLTransformer._parse_number"""

    def test_parse_number_decimal(self):
        """Test parsing decimal numbers."""
        assert DSLTransformer._parse_number("42") == 42
        assert DSLTransformer._parse_number("0") == 0
        assert DSLTransformer._parse_number("-10") == -10

    def test_parse_number_hexadecimal(self):
        """Test parsing hexadecimal numbers."""
        assert DSLTransformer._parse_number("0xFF") == 255
        assert DSLTransformer._parse_number("0x10") == 16
        assert DSLTransformer._parse_number("0XAB") == 171

    def test_parse_number_octal(self):
        """Test parsing octal numbers."""
        assert DSLTransformer._parse_number("0o755") == 493
        assert DSLTransformer._parse_number("0O10") == 8

    def test_parse_number_binary(self):
        """Test parsing binary numbers."""
        assert DSLTransformer._parse_number("0b1010") == 10
        assert DSLTransformer._parse_number("0B1111") == 15


class TestDSLTransformerSetPosition:
    """Tests for DSLTransformer._set_position"""

    def test_set_position_from_token(self):
        """Test setting position from a Token."""
        transformer = DSLTransformer()
        token = Token("NUMBER", "42")
        token.line = 10
        token.column = 5

        node = NumberLiteral(value=42, raw_text="42")
        result = transformer._set_position(node, token)

        assert result.line == 10
        assert result.column == 5

    def test_set_position_from_tree_meta(self):
        """Test setting position from a Tree with meta.

        Note: This test is skipped because Tree.meta is set internally by Lark
        and cannot be directly assigned in tests. The functionality is tested
        indirectly through parser integration tests.
        """
        pytest.skip("Cannot set Tree.meta externally - tested via integration")


class TestDSLTransformerStart:
    """Tests for DSLTransformer.start"""

    def test_start(self):
        """Test start rule creates Program node."""
        transformer = DSLTransformer()
        stmt = DeclStmt(type_name="Signal", name="x", value=NumberLiteral(value=42, raw_text="42"))
        result = transformer.start([stmt])

        assert isinstance(result, Program)
        assert len(result.statements) == 1
        assert result.statements[0] == stmt


class TestDSLTransformerDeclStmt:
    """Tests for DSLTransformer.decl_stmt"""

    def test_decl_stmt_full(self):
        """Test decl_stmt with all components."""
        transformer = DSLTransformer()
        type_token = Token("INT_KW", "Int")
        name_token = Token("NAME", "x")
        value = NumberLiteral(value=42, raw_text="42")

        result = transformer.decl_stmt([type_token, name_token, value])

        assert isinstance(result, DeclStmt)
        assert result.type_name == "Int"
        assert result.name == "x"
        assert result.value == value

    def test_decl_stmt_already_statement(self):
        """Test decl_stmt returns existing Statement."""
        transformer = DSLTransformer()
        stmt = DeclStmt(type_name="Signal", name="x", value=NumberLiteral(value=42, raw_text="42"))
        result = transformer.decl_stmt([stmt])

        assert result is stmt


class TestDSLTransformerMemDecl:
    """Tests for DSLTransformer.mem_decl"""

    def test_mem_decl_name_only(self):
        """Test mem_decl with only a name."""
        transformer = DSLTransformer()
        name_token = Token("NAME", "my_mem")

        result = transformer.mem_decl([name_token])

        assert isinstance(result, MemDecl)
        assert result.name == "my_mem"
        assert result.signal_type is None

    def test_mem_decl_with_signal_type(self):
        """Test mem_decl with signal type."""
        transformer = DSLTransformer()
        name_token = Token("NAME", "my_mem")
        type_token = Token("STRING", '"item"')

        result = transformer.mem_decl([name_token, type_token])

        assert isinstance(result, MemDecl)
        assert result.name == "my_mem"
        assert result.signal_type == "item"

    def test_mem_decl_already_mem_decl(self):
        """Test mem_decl returns existing MemDecl."""
        transformer = DSLTransformer()
        mem = MemDecl(name="existing", signal_type=None)
        result = transformer.mem_decl([mem])

        assert result is mem


class TestDSLTransformerTypeName:
    """Tests for DSLTransformer.type_name"""

    def test_type_name(self):
        """Test type_name returns token value."""
        transformer = DSLTransformer()
        token = Token("INT_KW", "Int")
        result = transformer.type_name([token])

        assert result == "Int"


class TestDSLTransformerUnwrapTree:
    """Tests for DSLTransformer._unwrap_tree"""

    def test_unwrap_tree_single_child(self):
        """Test unwrapping Tree with single child."""
        transformer = DSLTransformer()
        child = NumberLiteral(value=42, raw_text="42")
        tree = Tree("test", [child])

        result = transformer._unwrap_tree(tree)
        assert result is child

    def test_unwrap_tree_multiple_children(self):
        """Test unwrapping Tree with multiple children."""
        transformer = DSLTransformer()
        child1 = NumberLiteral(value=1, raw_text="1")
        child2 = NumberLiteral(value=2, raw_text="2")
        tree = Tree("test", [child1, child2])

        result = transformer._unwrap_tree(tree)
        assert result == [child1, child2]

    def test_unwrap_tree_token(self):
        """Test unwrapping Token converts to literal."""
        transformer = DSLTransformer()
        token = Token("NUMBER", "42")

        result = transformer._unwrap_tree(token)
        # Token gets converted to a NumberLiteral
        assert isinstance(result, NumberLiteral)
        assert result.value == 42


class TestDSLTransformerAssignStmt:
    """Tests for DSLTransformer.assign_stmt"""

    def test_assign_stmt(self):
        """Test assign_stmt creates AssignStmt."""
        transformer = DSLTransformer()
        lvalue = Identifier(name="x", raw_text="x")
        value = NumberLiteral(value=42, raw_text="42")

        result = transformer.assign_stmt([lvalue, value])

        assert isinstance(result, AssignStmt)
        assert result.target == lvalue
        assert result.value == value


class TestDSLTransformerExprStmt:
    """Tests for DSLTransformer.expr_stmt"""

    def test_expr_stmt(self):
        """Test expr_stmt creates ExprStmt."""
        transformer = DSLTransformer()
        expr = NumberLiteral(value=42, raw_text="42")

        result = transformer.expr_stmt([expr])

        assert isinstance(result, ExprStmt)
        assert result.expr == expr


class TestDSLTransformerStatementExprStmt:
    """Tests for DSLTransformer.statement_expr_stmt"""

    def test_statement_expr_stmt(self):
        """Test statement_expr_stmt returns first item."""
        transformer = DSLTransformer()
        expr = NumberLiteral(value=42, raw_text="42")

        # statement_expr_stmt just returns items[0] to avoid double nesting
        result = transformer.statement_expr_stmt([expr])

        assert result is expr


class TestDSLTransformerReturnStmt:
    """Tests for DSLTransformer.return_stmt"""

    def test_return_stmt_with_value(self):
        """Test return_stmt with return value."""
        transformer = DSLTransformer()
        value = NumberLiteral(value=42, raw_text="42")

        result = transformer.return_stmt([value])

        assert isinstance(result, ReturnStmt)
        assert result.expr == value

    def test_return_stmt_no_value(self):
        """Test return_stmt without return value."""
        transformer = DSLTransformer()
        # When there's no value, a None or default value is passed
        result = transformer.return_stmt([None])

        assert isinstance(result, ReturnStmt)
        assert result.expr is None


class TestDSLTransformerImportStmt:
    """Tests for DSLTransformer.import_stmt"""

    def test_import_stmt(self):
        """Test import_stmt creates ImportStmt."""
        transformer = DSLTransformer()
        path_token = Token("STRING", '"lib/mylib.facto"')

        result = transformer.import_stmt([path_token])

        assert isinstance(result, ImportStmt)
        assert result.path == "lib/mylib.facto"


class TestDSLTransformerFuncDecl:
    """Tests for DSLTransformer.func_decl"""

    def test_func_decl_no_params(self):
        """Test func_decl without parameters."""
        transformer = DSLTransformer()
        name_token = Token("NAME", "my_func")
        body_stmt = ReturnStmt(expr=NumberLiteral(value=42, raw_text="42"))

        result = transformer.func_decl([name_token, body_stmt])

        assert isinstance(result, FuncDecl)
        assert result.name == "my_func"
        assert len(result.params) == 0
        assert len(result.body) == 1

    def test_func_decl_with_params(self):
        """Test func_decl with parameters."""
        transformer = DSLTransformer()
        name_token = Token("NAME", "add")
        params = [TypedParam(type_name="Int", name="a"), TypedParam(type_name="Int", name="b")]
        body_stmt = ReturnStmt(expr=NumberLiteral(value=0, raw_text="0"))

        result = transformer.func_decl([name_token, params, body_stmt])

        assert isinstance(result, FuncDecl)
        assert result.name == "add"
        assert len(result.params) == 2
        assert result.params[0].name == "a"
        assert result.params[1].name == "b"


class TestDSLTransformerTypedParamList:
    """Tests for DSLTransformer.typed_param_list"""

    def test_typed_param_list(self):
        """Test typed_param_list returns list of TypedParam."""
        transformer = DSLTransformer()
        param1 = TypedParam(type_name="Int", name="x")
        param2 = TypedParam(type_name="Signal", name="y")

        result = transformer.typed_param_list([param1, param2])

        assert len(result) == 2
        assert result[0] == param1
        assert result[1] == param2


class TestDSLTransformerTypedParam:
    """Tests for DSLTransformer.typed_param"""

    def test_typed_param(self):
        """Test typed_param creates TypedParam."""
        transformer = DSLTransformer()
        type_token = Token("INT_KW", "Int")
        name_token = Token("NAME", "x")

        result = transformer.typed_param([type_token, name_token])

        assert isinstance(result, TypedParam)
        assert result.type_name == "Int"
        assert result.name == "x"


class TestDSLTransformerForStmt:
    """Tests for DSLTransformer.for_stmt"""

    def test_for_stmt_range_iterator(self):
        """Test for_stmt with range iterator."""
        transformer = DSLTransformer()
        var_token = Token("NAME", "i")
        iterator = {"start": 0, "stop": 10, "step": 1}
        body_stmt = ExprStmt(expr=NumberLiteral(value=1, raw_text="1"))

        result = transformer.for_stmt([var_token, iterator, body_stmt])

        assert isinstance(result, ForStmt)
        assert result.iterator_name == "i"
        assert result.start == 0
        assert result.stop == 10
        assert result.step == 1
        assert len(result.body) == 1

    def test_for_stmt_list_iterator(self):
        """Test for_stmt with list iterator."""
        transformer = DSLTransformer()
        var_token = Token("NAME", "i")
        iterator = {"values": [1, 2, 3]}
        body_stmt = ExprStmt(expr=NumberLiteral(value=1, raw_text="1"))

        result = transformer.for_stmt([var_token, iterator, body_stmt])

        assert isinstance(result, ForStmt)
        assert result.iterator_name == "i"
        assert result.values == [1, 2, 3]


class TestDSLTransformerForIterator:
    """Tests for DSLTransformer.for_iterator"""

    def test_for_iterator(self):
        """Test for_iterator returns first item."""
        transformer = DSLTransformer()
        iterator_dict = {"start": 0, "stop": 10}

        result = transformer.for_iterator([iterator_dict])

        assert result == iterator_dict


class TestDSLTransformerRangeBound:
    """Tests for DSLTransformer.range_bound"""

    def test_range_bound_number(self):
        """Test range_bound with NUMBER token."""
        transformer = DSLTransformer()
        token = Token("NUMBER", "42")

        result = transformer.range_bound([token])

        assert result == 42

    def test_range_bound_variable(self):
        """Test range_bound with NAME token."""
        transformer = DSLTransformer()
        token = Token("NAME", "max_val")

        result = transformer.range_bound([token])

        assert result == "max_val"


class TestDSLTransformerRangeIterator:
    """Tests for DSLTransformer.range_iterator"""

    def test_range_iterator_simple(self):
        """Test range_iterator with start and stop."""
        transformer = DSLTransformer()
        result = transformer.range_iterator([0, Token("RANGE_OP", ".."), 10])

        assert result["start"] == 0
        assert result["stop"] == 10
        assert result["step"] == 1

    def test_range_iterator_with_step(self):
        """Test range_iterator with start, stop, and step."""
        transformer = DSLTransformer()
        result = transformer.range_iterator(
            [0, Token("RANGE_OP", ".."), 10, Token("STEP_KW", "step"), 2]
        )

        assert result["start"] == 0
        assert result["stop"] == 10
        assert result["step"] == 2


class TestDSLTransformerListIterator:
    """Tests for DSLTransformer.list_iterator"""

    def test_list_iterator(self):
        """Test list_iterator creates values list."""
        transformer = DSLTransformer()
        num1 = Token("NUMBER", "1")
        num2 = Token("NUMBER", "2")
        num3 = Token("NUMBER", "3")

        result = transformer.list_iterator([num1, num2, num3])

        assert result["values"] == [1, 2, 3]


class TestDSLTransformerParamType:
    """Tests for DSLTransformer.param_type"""

    def test_param_type(self):
        """Test param_type returns type string."""
        transformer = DSLTransformer()
        token = Token("INT_KW", "Int")

        result = transformer.param_type([token])

        assert result == "Int"


class TestDSLTransformerLValue:
    """Tests for DSLTransformer.lvalue"""

    def test_lvalue_identifier(self):
        """Test lvalue creates Identifier."""
        transformer = DSLTransformer()
        token = Token("NAME", "x")

        result = transformer.lvalue([token])

        assert isinstance(result, Identifier)
        assert result.name == "x"

    def test_lvalue_property_access(self):
        """Test lvalue creates PropertyAccess."""
        transformer = DSLTransformer()
        obj_token = Token("NAME", "entity")
        prop_token = Token("NAME", "enabled")

        result = transformer.lvalue([obj_token, prop_token])

        assert isinstance(result, PropertyAccess)
        assert result.object_name == "entity"
        assert result.property_name == "enabled"


class TestDSLTransformerExpr:
    """Tests for DSLTransformer.expr"""

    def test_expr(self):
        """Test expr returns first item."""
        transformer = DSLTransformer()
        expr = NumberLiteral(value=42, raw_text="42")

        result = transformer.expr([expr])

        assert result is expr


class TestDSLTransformerLogicOr:
    """Tests for DSLTransformer.logic_or"""

    def test_logic_or_single(self):
        """Test logic_or with single operand."""
        transformer = DSLTransformer()
        expr = NumberLiteral(value=1, raw_text="1")

        result = transformer.logic_or([expr])

        assert result is expr

    def test_logic_or_chain(self):
        """Test logic_or with multiple operands."""
        transformer = DSLTransformer()
        left = NumberLiteral(value=1, raw_text="1")
        right = NumberLiteral(value=2, raw_text="2")

        result = transformer.logic_or([left, Token("LOGIC_OR", "or"), right])

        assert isinstance(result, BinaryOp)
        assert result.op == "||"
        assert result.left == left
        assert result.right == right


class TestDSLTransformerLogicAnd:
    """Tests for DSLTransformer.logic_and"""

    def test_logic_and_single(self):
        """Test logic_and with single operand."""
        transformer = DSLTransformer()
        expr = NumberLiteral(value=1, raw_text="1")

        result = transformer.logic_and([expr])

        assert result is expr

    def test_logic_and_chain(self):
        """Test logic_and with multiple operands."""
        transformer = DSLTransformer()
        left = NumberLiteral(value=1, raw_text="1")
        right = NumberLiteral(value=2, raw_text="2")

        result = transformer.logic_and([left, Token("LOGIC_AND", "and"), right])

        assert isinstance(result, BinaryOp)
        assert result.op == "&&"
        assert result.left == left
        assert result.right == right


class TestDSLTransformerOutputSpec:
    """Tests for DSLTransformer.output_spec"""

    def test_output_spec_no_output(self):
        """Test output_spec without output specifier."""
        transformer = DSLTransformer()
        comparison = NumberLiteral(value=1, raw_text="1")

        result = transformer.output_spec([comparison])

        assert result is comparison

    def test_output_spec_with_output(self):
        """Test output_spec with output specifier."""
        transformer = DSLTransformer()
        condition = NumberLiteral(value=1, raw_text="1")
        output = NumberLiteral(value=2, raw_text="2")

        result = transformer.output_spec([condition, Token("OUTPUT_SPEC", ":"), output])

        assert isinstance(result, OutputSpecExpr)
        assert result.condition == condition
        assert result.output_value == output


class TestDSLTransformerOutputValue:
    """Tests for DSLTransformer.output_value"""

    def test_output_value(self):
        """Test output_value returns expression."""
        transformer = DSLTransformer()
        expr = NumberLiteral(value=42, raw_text="42")

        result = transformer.output_value([expr])

        assert result is expr


class TestDSLTransformerComparison:
    """Tests for DSLTransformer.comparison"""

    def test_comparison_single(self):
        """Test comparison with single operand."""
        transformer = DSLTransformer()
        expr = NumberLiteral(value=1, raw_text="1")

        result = transformer.comparison([expr])

        assert result is expr

    def test_comparison_chain(self):
        """Test comparison with operator."""
        transformer = DSLTransformer()
        left = NumberLiteral(value=1, raw_text="1")
        right = NumberLiteral(value=2, raw_text="2")

        result = transformer.comparison([left, Token("COMP_OP", "=="), right])

        assert isinstance(result, BinaryOp)
        assert result.op == "=="


class TestDSLTransformerProjection:
    """Tests for DSLTransformer.projection"""

    def test_projection_single(self):
        """Test projection with single expression."""
        transformer = DSLTransformer()
        expr = NumberLiteral(value=1, raw_text="1")

        result = transformer.projection([expr])

        assert result is expr

    def test_projection_with_type(self):
        """Test projection with type cast."""
        transformer = DSLTransformer()
        expr = NumberLiteral(value=1, raw_text="1")

        result = transformer.projection([expr, Token("PROJ_OP", "::"), "item"])

        assert isinstance(result, ProjectionExpr)
        assert result.expr == expr
        assert result.target_type == "item"


class TestDSLTransformerBitwiseOr:
    """Tests for DSLTransformer.bitwise_or"""

    def test_bitwise_or_single(self):
        """Test bitwise_or with single operand."""
        transformer = DSLTransformer()
        expr = NumberLiteral(value=1, raw_text="1")

        result = transformer.bitwise_or([expr])

        assert result is expr

    def test_bitwise_or_chain(self):
        """Test bitwise_or with operator."""
        transformer = DSLTransformer()
        left = NumberLiteral(value=1, raw_text="1")
        right = NumberLiteral(value=2, raw_text="2")

        result = transformer.bitwise_or([left, Token("BITWISE_OR", "|"), right])

        assert isinstance(result, BinaryOp)
        assert result.op == "|"


class TestDSLTransformerBitwiseXor:
    """Tests for DSLTransformer.bitwise_xor"""

    def test_bitwise_xor_single(self):
        """Test bitwise_xor with single operand."""
        transformer = DSLTransformer()
        expr = NumberLiteral(value=1, raw_text="1")

        result = transformer.bitwise_xor([expr])

        assert result is expr

    def test_bitwise_xor_chain(self):
        """Test bitwise_xor with operator."""
        transformer = DSLTransformer()
        left = NumberLiteral(value=1, raw_text="1")
        right = NumberLiteral(value=2, raw_text="2")

        result = transformer.bitwise_xor([left, Token("BITWISE_XOR", "^"), right])

        assert isinstance(result, BinaryOp)
        assert result.op == "^"


class TestDSLTransformerBitwiseAnd:
    """Tests for DSLTransformer.bitwise_and"""

    def test_bitwise_and_single(self):
        """Test bitwise_and with single operand."""
        transformer = DSLTransformer()
        expr = NumberLiteral(value=1, raw_text="1")

        result = transformer.bitwise_and([expr])

        assert result is expr

    def test_bitwise_and_chain(self):
        """Test bitwise_and with operator."""
        transformer = DSLTransformer()
        left = NumberLiteral(value=1, raw_text="1")
        right = NumberLiteral(value=2, raw_text="2")

        result = transformer.bitwise_and([left, Token("BITWISE_AND", "&"), right])

        assert isinstance(result, BinaryOp)
        assert result.op == "&"


class TestDSLTransformerShift:
    """Tests for DSLTransformer.shift"""

    def test_shift_single(self):
        """Test shift with single operand."""
        transformer = DSLTransformer()
        expr = NumberLiteral(value=1, raw_text="1")

        result = transformer.shift([expr])

        assert result is expr

    def test_shift_chain(self):
        """Test shift with operator."""
        transformer = DSLTransformer()
        left = NumberLiteral(value=1, raw_text="1")
        right = NumberLiteral(value=2, raw_text="2")

        result = transformer.shift([left, Token("SHIFT_OP", "<<"), right])

        assert isinstance(result, BinaryOp)
        assert result.op == "<<"


class TestDSLTransformerAdd:
    """Tests for DSLTransformer.add"""

    def test_add_single(self):
        """Test add with single operand."""
        transformer = DSLTransformer()
        expr = NumberLiteral(value=1, raw_text="1")

        result = transformer.add([expr])

        assert result is expr

    def test_add_chain(self):
        """Test add with operator."""
        transformer = DSLTransformer()
        left = NumberLiteral(value=1, raw_text="1")
        right = NumberLiteral(value=2, raw_text="2")

        result = transformer.add([left, Token("ADD_OP", "+"), right])

        assert isinstance(result, BinaryOp)
        assert result.op == "+"


class TestDSLTransformerMul:
    """Tests for DSLTransformer.mul"""

    def test_mul_single(self):
        """Test mul with single operand."""
        transformer = DSLTransformer()
        expr = NumberLiteral(value=1, raw_text="1")

        result = transformer.mul([expr])

        assert result is expr

    def test_mul_chain(self):
        """Test mul with operator."""
        transformer = DSLTransformer()
        left = NumberLiteral(value=2, raw_text="2")
        right = NumberLiteral(value=3, raw_text="3")

        result = transformer.mul([left, Token("MUL_OP", "*"), right])

        assert isinstance(result, BinaryOp)
        assert result.op == "*"


class TestDSLTransformerPower:
    """Tests for DSLTransformer.power"""

    def test_power_single(self):
        """Test power with single operand."""
        transformer = DSLTransformer()
        expr = NumberLiteral(value=2, raw_text="2")

        result = transformer.power([expr])

        assert result is expr

    def test_power_chain(self):
        """Test power with operator."""
        transformer = DSLTransformer()
        left = NumberLiteral(value=2, raw_text="2")
        left.line = 1
        left.column = 0
        right = NumberLiteral(value=3, raw_text="3")

        result = transformer.power([left, Token("POWER_OP", "**"), right])

        assert isinstance(result, BinaryOp)
        assert result.op == "**"
        assert result.left == left
        assert result.right == right


class TestDSLTransformerUnary:
    """Tests for DSLTransformer.unary"""

    def test_unary_single(self):
        """Test unary with no operator."""
        transformer = DSLTransformer()
        expr = NumberLiteral(value=42, raw_text="42")

        result = transformer.unary([expr])

        assert result is expr

    def test_unary_negation(self):
        """Test unary negation operator."""
        transformer = DSLTransformer()
        expr = NumberLiteral(value=42, raw_text="42")

        result = transformer.unary([Token("OP", "-"), expr])

        assert isinstance(result, UnaryOp)
        assert result.op == "-"
        assert result.expr == expr

    def test_unary_not(self):
        """Test unary not operator."""
        transformer = DSLTransformer()
        expr = NumberLiteral(value=1, raw_text="1")

        result = transformer.unary([Token("OP", "!"), expr])

        assert isinstance(result, UnaryOp)
        assert result.op == "!"


class TestDSLTransformerNormalizeLogicalOp:
    """Tests for DSLTransformer._normalize_logical_op"""

    def test_normalize_logical_op_and(self):
        """Test normalizing 'and' to '&&'."""
        transformer = DSLTransformer()
        assert transformer._normalize_logical_op("and") == "&&"
        assert transformer._normalize_logical_op("AND") == "&&"

    def test_normalize_logical_op_or(self):
        """Test normalizing 'or' to '||'."""
        transformer = DSLTransformer()
        assert transformer._normalize_logical_op("or") == "||"
        assert transformer._normalize_logical_op("OR") == "||"

    def test_normalize_logical_op_other(self):
        """Test other operators pass through."""
        transformer = DSLTransformer()
        assert transformer._normalize_logical_op("+") == "+"


class TestDSLTransformerHandleBinaryOpChain:
    """Tests for DSLTransformer._handle_binary_op_chain"""

    def test_handle_binary_op_chain_single(self):
        """Test binary op chain with single item."""
        transformer = DSLTransformer()
        expr = NumberLiteral(value=1, raw_text="1")

        result = transformer._handle_binary_op_chain([expr])

        assert result is expr

    def test_handle_binary_op_chain_simple(self):
        """Test binary op chain with two operands."""
        transformer = DSLTransformer()
        left = NumberLiteral(value=1, raw_text="1")
        left.line = 1
        left.column = 0
        right = NumberLiteral(value=2, raw_text="2")

        result = transformer._handle_binary_op_chain([left, Token("ADD_OP", "+"), right])

        assert isinstance(result, BinaryOp)
        assert result.op == "+"
        assert result.left == left
        assert result.right == right

    def test_handle_binary_op_chain_multiple(self):
        """Test binary op chain with multiple operations."""
        transformer = DSLTransformer()
        a = NumberLiteral(value=1, raw_text="1")
        a.line = 1
        b = NumberLiteral(value=2, raw_text="2")
        c = NumberLiteral(value=3, raw_text="3")

        result = transformer._handle_binary_op_chain(
            [a, Token("ADD_OP", "+"), b, Token("ADD_OP", "+"), c]
        )

        assert isinstance(result, BinaryOp)
        # Should create ((a + b) + c)
        assert result.right == c
        assert isinstance(result.left, BinaryOp)


class TestDSLTransformerCallExpr:
    """Tests for DSLTransformer.call_expr"""

    def test_call_expr_no_args(self):
        """Test call_expr without arguments."""
        transformer = DSLTransformer()
        name_token = Token("NAME", "my_func")

        result = transformer.call_expr([name_token])

        assert isinstance(result, CallExpr)
        assert result.name == "my_func"
        assert len(result.args) == 0

    def test_call_expr_with_args(self):
        """Test call_expr with arguments."""
        transformer = DSLTransformer()
        name_token = Token("NAME", "add")
        arg1 = NumberLiteral(value=1, raw_text="1")
        arg2 = NumberLiteral(value=2, raw_text="2")

        result = transformer.call_expr([name_token, [arg1, arg2]])

        assert isinstance(result, CallExpr)
        assert result.name == "add"
        assert len(result.args) == 2


class TestDSLTransformerMethodCall:
    """Tests for DSLTransformer.method_call"""

    def test_method_call_no_args(self):
        """Test method_call without arguments."""
        transformer = DSLTransformer()
        module_token = Token("NAME", "math")
        method_token = Token("NAME", "abs")

        result = transformer.method_call([module_token, method_token])

        assert isinstance(result, CallExpr)
        assert result.name == "math.abs"
        assert len(result.args) == 0

    def test_method_call_with_args(self):
        """Test method_call with arguments."""
        transformer = DSLTransformer()
        module_token = Token("NAME", "math")
        method_token = Token("NAME", "max")
        arg1 = NumberLiteral(value=1, raw_text="1")
        arg2 = NumberLiteral(value=2, raw_text="2")

        result = transformer.method_call([module_token, method_token, [arg1, arg2]])

        assert isinstance(result, CallExpr)
        assert result.name == "math.max"
        assert len(result.args) == 2


class TestDSLTransformerMemoryRead:
    """Tests for DSLTransformer.memory_read"""

    def test_memory_read(self):
        """Test memory_read creates ReadExpr."""
        transformer = DSLTransformer()
        mem_token = Token("NAME", "my_mem")

        result = transformer.memory_read([mem_token])

        assert isinstance(result, ReadExpr)
        assert result.memory_name == "my_mem"


class TestDSLTransformerMemoryWrite:
    """Tests for DSLTransformer.memory_write"""

    def test_memory_write(self):
        """Test memory_write creates WriteExpr."""
        transformer = DSLTransformer()
        mem_token = Token("NAME", "my_mem")
        value = NumberLiteral(value=42, raw_text="42")

        result = transformer.memory_write([mem_token, value])

        assert isinstance(result, WriteExpr)
        assert result.memory_name == "my_mem"
        assert result.value == value
        assert result.when is None


class TestDSLTransformerMemoryWriteWhen:
    """Tests for DSLTransformer.memory_write_when"""

    def test_memory_write_when(self):
        """Test memory_write_when creates WriteExpr with condition."""
        transformer = DSLTransformer()
        mem_token = Token("NAME", "my_mem")
        value = NumberLiteral(value=42, raw_text="42")
        condition = NumberLiteral(value=1, raw_text="1")

        result = transformer.memory_write_when(
            [mem_token, value, Token("WHEN_KW", "when"), condition]
        )

        assert isinstance(result, WriteExpr)
        assert result.memory_name == "my_mem"
        assert result.value == value
        assert result.when == condition


class TestDSLTransformerMemoryLatchWrite:
    """Tests for DSLTransformer.memory_latch_write"""

    def test_memory_latch_write(self):
        """Test memory_latch_write creates WriteExpr with latch signals."""
        transformer = DSLTransformer()
        mem_token = Token("NAME", "my_mem")
        value = NumberLiteral(value=1, raw_text="1")
        set_expr = NumberLiteral(value=1, raw_text="1")
        reset_expr = NumberLiteral(value=0, raw_text="0")

        result = transformer.memory_latch_write([mem_token, value, (set_expr, reset_expr, True)])

        assert isinstance(result, WriteExpr)
        assert result.memory_name == "my_mem"
        assert result.value == value
        assert result.set_signal == set_expr
        assert result.reset_signal == reset_expr
        assert result.set_priority is True


class TestDSLTransformerLatchSetReset:
    """Tests for DSLTransformer.latch_set_reset"""

    def test_latch_set_reset(self):
        """Test latch_set_reset returns SR latch tuple."""
        transformer = DSLTransformer()
        set_expr = NumberLiteral(value=1, raw_text="1")
        reset_expr = NumberLiteral(value=0, raw_text="0")

        result = transformer.latch_set_reset(
            [Token("SET_KW", "set"), set_expr, Token("RESET_KW", "reset"), reset_expr]
        )

        assert result == (set_expr, reset_expr, True)


class TestDSLTransformerLatchResetSet:
    """Tests for DSLTransformer.latch_reset_set"""

    def test_latch_reset_set(self):
        """Test latch_reset_set returns RS latch tuple."""
        transformer = DSLTransformer()
        set_expr = NumberLiteral(value=1, raw_text="1")
        reset_expr = NumberLiteral(value=0, raw_text="0")

        result = transformer.latch_reset_set(
            [Token("RESET_KW", "reset"), reset_expr, Token("SET_KW", "set"), set_expr]
        )

        assert result == (set_expr, reset_expr, False)


class TestDSLTransformerArglist:
    """Tests for DSLTransformer.arglist"""

    def test_arglist(self):
        """Test arglist returns list of expressions."""
        transformer = DSLTransformer()
        arg1 = NumberLiteral(value=1, raw_text="1")
        arg2 = NumberLiteral(value=2, raw_text="2")

        result = transformer.arglist([arg1, arg2])

        assert result == [arg1, arg2]


class TestDSLTransformerDictLiteral:
    """Tests for DSLTransformer.dict_literal"""

    def test_dict_literal_empty(self):
        """Test dict_literal with no items."""
        transformer = DSLTransformer()
        result = transformer.dict_literal([])

        assert isinstance(result, DictLiteral)
        assert len(result.entries) == 0

    def test_dict_literal_with_items(self):
        """Test dict_literal with items."""
        transformer = DSLTransformer()
        val1 = NumberLiteral(value=1, raw_text="1")
        val2 = NumberLiteral(value=2, raw_text="2")
        token = Token("STRING", '"key1"')

        result = transformer.dict_literal([("key1", val1, token), ("key2", val2, token)])

        assert isinstance(result, DictLiteral)
        assert len(result.entries) == 2
        assert result.entries["key1"] == val1
        assert result.entries["key2"] == val2


class TestDSLTransformerDictItem:
    """Tests for DSLTransformer.dict_item"""

    def test_dict_item_string_key(self):
        """Test dict_item with string key."""
        transformer = DSLTransformer()
        key_token = Token("STRING", '"mykey"')
        value = NumberLiteral(value=42, raw_text="42")

        key, val, token = transformer.dict_item([key_token, value])

        assert key == "mykey"
        assert val == value

    def test_dict_item_name_key(self):
        """Test dict_item with NAME key."""
        transformer = DSLTransformer()
        key_token = Token("NAME", "mykey")
        value = NumberLiteral(value=42, raw_text="42")

        key, val, token = transformer.dict_item([key_token, value])

        assert key == "mykey"
        assert val == value


class TestDSLTransformerEmptyBrace:
    """Tests for DSLTransformer.empty_brace"""

    def test_empty_brace(self):
        """Test empty_brace creates empty bundle."""
        transformer = DSLTransformer()
        result = transformer.empty_brace([])

        assert isinstance(result, BundleLiteral)
        assert len(result.elements) == 0


class TestDSLTransformerBundleLiteral:
    """Tests for DSLTransformer.bundle_literal"""

    def test_bundle_literal_empty(self):
        """Test bundle_literal with no elements."""
        transformer = DSLTransformer()
        result = transformer.bundle_literal([])

        assert isinstance(result, BundleLiteral)
        assert len(result.elements) == 0

    def test_bundle_literal_with_elements(self):
        """Test bundle_literal with elements."""
        transformer = DSLTransformer()
        elem1 = NumberLiteral(value=1, raw_text="1")
        elem2 = NumberLiteral(value=2, raw_text="2")

        result = transformer.bundle_literal([elem1, elem2])

        assert isinstance(result, BundleLiteral)
        assert len(result.elements) == 2


class TestDSLTransformerBundleElement:
    """Tests for DSLTransformer.bundle_element"""

    def test_bundle_element(self):
        """Test bundle_element returns expression."""
        transformer = DSLTransformer()
        expr = NumberLiteral(value=42, raw_text="42")

        result = transformer.bundle_element([expr])

        assert result == expr


class TestDSLTransformerBundleSelect:
    """Tests for DSLTransformer.bundle_select"""

    def test_bundle_select(self):
        """Test bundle_select creates BundleSelectExpr."""
        transformer = DSLTransformer()
        bundle_token = Token("NAME", "my_bundle")
        type_token = Token("STRING", '"item"')

        result = transformer.bundle_select([bundle_token, type_token])

        assert isinstance(result, BundleSelectExpr)
        assert result.bundle.name == "my_bundle"
        assert result.signal_type == "item"


class TestDSLTransformerBundleAny:
    """Tests for DSLTransformer.bundle_any"""

    def test_bundle_any(self):
        """Test bundle_any creates BundleAnyExpr."""
        transformer = DSLTransformer()
        any_token = Token("ANY_KW", "any")
        any_token.line = 1
        bundle = IdentifierExpr(name="my_bundle")

        result = transformer.bundle_any([any_token, bundle])

        assert isinstance(result, BundleAnyExpr)
        assert result.bundle == bundle


class TestDSLTransformerBundleAll:
    """Tests for DSLTransformer.bundle_all"""

    def test_bundle_all(self):
        """Test bundle_all creates BundleAllExpr."""
        transformer = DSLTransformer()
        all_token = Token("ALL_KW", "all")
        all_token.line = 1
        bundle = IdentifierExpr(name="my_bundle")

        result = transformer.bundle_all([all_token, bundle])

        assert isinstance(result, BundleAllExpr)
        assert result.bundle == bundle


class TestDSLTransformerSignalWithType:
    """Tests for DSLTransformer.signal_with_type"""

    def test_signal_with_type(self):
        """Test signal_with_type creates SignalLiteral with type."""
        transformer = DSLTransformer()
        value = NumberLiteral(value=42, raw_text="42")

        result = transformer.signal_with_type(["item", value])

        assert isinstance(result, SignalLiteral)
        assert result.signal_type == "item"
        assert result.value == value


class TestDSLTransformerSignalConstant:
    """Tests for DSLTransformer.signal_constant"""

    def test_signal_constant(self):
        """Test signal_constant creates SignalLiteral without type."""
        transformer = DSLTransformer()
        token = Token("NUMBER", "42")

        result = transformer.signal_constant([token])

        assert isinstance(result, SignalLiteral)
        assert result.signal_type is None
        assert isinstance(result.value, NumberLiteral)


class TestDSLTransformerTypePropertyAccess:
    """Tests for DSLTransformer.type_property_access"""

    def test_type_property_access(self):
        """Test type_property_access creates SignalTypeAccess."""
        transformer = DSLTransformer()
        obj_token = Token("NAME", "my_signal")
        prop_token = Token("NAME", "type")

        result = transformer.type_property_access([obj_token, prop_token])

        assert isinstance(result, SignalTypeAccess)
        assert result.object_name == "my_signal"
        assert result.property_name == "type"


class TestDSLTransformerTypeLiteral:
    """Tests for DSLTransformer.type_literal"""

    def test_type_literal_string(self):
        """Test type_literal with string token."""
        transformer = DSLTransformer()
        token = Token("STRING", '"item"')

        result = transformer.type_literal([token])

        assert result == "item"

    def test_type_literal_name(self):
        """Test type_literal with NAME token."""
        transformer = DSLTransformer()
        token = Token("NAME", "fluid")

        result = transformer.type_literal([token])

        assert result == "fluid"

    def test_type_literal_signal_type_access(self):
        """Test type_literal with SignalTypeAccess."""
        transformer = DSLTransformer()
        access = SignalTypeAccess(object_name="sig", property_name="type")

        result = transformer.type_literal([access])

        assert result is access


class TestDSLTransformerPrimary:
    """Tests for DSLTransformer.primary"""

    def test_primary_expr(self):
        """Test primary returns Expr as-is."""
        transformer = DSLTransformer()
        expr = NumberLiteral(value=42, raw_text="42")

        result = transformer.primary([expr])

        assert result is expr

    def test_primary_name_token(self):
        """Test primary converts NAME token to IdentifierExpr."""
        transformer = DSLTransformer()
        token = Token("NAME", "my_var")

        result = transformer.primary([token])

        assert isinstance(result, IdentifierExpr)
        assert result.name == "my_var"

    def test_primary_identifier(self):
        """Test primary converts Identifier to IdentifierExpr."""
        transformer = DSLTransformer()
        ident = Identifier(name="my_var", raw_text="my_var")

        result = transformer.primary([ident])

        assert isinstance(result, IdentifierExpr)
        assert result.name == "my_var"

    def test_primary_property_access_output(self):
        """Test primary converts PropertyAccess.output to EntityOutputExpr."""
        transformer = DSLTransformer()
        prop = PropertyAccess(
            object_name="my_entity", property_name="output", raw_text="my_entity.output"
        )

        result = transformer.primary([prop])

        assert isinstance(result, EntityOutputExpr)
        assert result.entity_name == "my_entity"

    def test_primary_property_access_other(self):
        """Test primary converts PropertyAccess to PropertyAccessExpr."""
        transformer = DSLTransformer()
        prop = PropertyAccess(
            object_name="my_entity", property_name="enabled", raw_text="my_entity.enabled"
        )

        result = transformer.primary([prop])

        assert isinstance(result, PropertyAccessExpr)
        assert result.object_name == "my_entity"
        assert result.property_name == "enabled"
