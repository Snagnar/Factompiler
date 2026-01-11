"""
Tests for semantic/analyzer.py - Semantic analysis functionality.
"""

from pathlib import Path

import pytest

from dsl_compiler.src.ast.expressions import SignalTypeAccess
from dsl_compiler.src.ast.literals import NumberLiteral
from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.common.signal_registry import SignalTypeInfo
from dsl_compiler.src.parsing.parser import DSLParser
from dsl_compiler.src.semantic.analyzer import SemanticAnalyzer
from dsl_compiler.src.semantic.symbol_table import SymbolType
from dsl_compiler.src.semantic.type_system import IntValue, SignalValue


class TestSemanticAnalyzer:
    """Tests for SemanticAnalyzer class."""

    @pytest.fixture
    def parser(self):
        """Create a new parser instance."""
        return DSLParser()

    @pytest.fixture
    def diagnostics(self):
        """Create a new diagnostics instance."""
        return ProgramDiagnostics()

    @pytest.fixture
    def analyzer(self, diagnostics):
        """Create a new analyzer instance."""
        return SemanticAnalyzer(diagnostics)

    def test_analyzer_initialization(self, analyzer):
        """Test analyzer can be initialized."""
        assert analyzer is not None

    def test_analyzer_has_symbol_table(self, analyzer):
        """Test analyzer has a symbol table."""
        assert analyzer.symbol_table is not None

    def test_analyzer_has_signal_registry(self, analyzer):
        """Test analyzer has a signal registry."""
        assert analyzer.signal_registry is not None


class TestSemanticAnalyzerVisit:
    """Tests for SemanticAnalyzer.visit() method."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    def test_visit_basic_declaration(self, parser):
        """Test visiting a basic signal declaration."""
        program = parser.parse("Signal x = 42;")
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert not diagnostics.has_errors()

    def test_visit_detects_undefined_variable(self, parser):
        """Test analyzer detects undefined variable usage."""
        program = parser.parse("Signal x = 5; Signal y = x + z;")
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert diagnostics.has_errors()


class TestSemanticAnalyzerLegacySyntax:
    """Tests for legacy syntax rejection."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    def test_write_legacy_syntax_rejected(self, parser):
        """Ensure legacy write(memory, value) form produces a migration error."""
        code = """
        Memory counter: "signal-A";
        Signal value = 42;
        write(counter, value);
        """
        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)

        assert diagnostics.has_errors(), "Legacy write syntax should raise an error"
        messages = diagnostics.get_messages()
        assert any("not a memory symbol" in msg for msg in messages)


class TestSemanticAnalyzerMemoryWrite:
    """Tests for memory.write() semantic analysis."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    def test_write_with_enable_signal_passes(self, parser):
        """Verify semantic analysis accepts memory.write(value, when=signal)."""
        code = """
        Memory counter: "signal-A";
        Signal enable = 1;
        counter.write(counter.read() + 1, when=enable);
        """
        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)

        assert not diagnostics.has_errors(), diagnostics.get_messages()


class TestSemanticAnalyzerSignalWReservation:
    """Tests for signal-W reservation enforcement."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    def test_signal_w_literal_is_rejected(self, parser):
        """User-declared signal literals must not target the reserved signal-W channel."""
        code = 'Signal bad = ("signal-W", 10);'
        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)

        assert diagnostics.has_errors(), "Expected an error when using reserved signal-W"
        messages = diagnostics.get_messages()
        assert any("signal-W" in msg and "reserved" in msg for msg in messages)

    def test_signal_w_projection_is_rejected(self, parser):
        """Projecting onto signal-W must surface a reservation error."""
        code = 'Signal x = 10 | "signal-W";'
        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)

        assert diagnostics.has_errors(), "Expected an error when projecting onto reserved signal-W"
        messages = diagnostics.get_messages()
        assert any("signal-W" in msg and "reserved" in msg for msg in messages)

    def test_signal_w_memory_declaration_is_rejected(self, parser):
        """Memory declarations must not claim the reserved signal-W channel."""
        code = 'Memory w: "signal-W";'
        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)

        assert diagnostics.has_errors()
        messages = diagnostics.get_messages()
        assert any("signal-W" in msg and "reserved" in msg for msg in messages)


class TestSemanticAnalyzerImplicitSignalAllocation:
    """Tests for implicit signal allocation."""

    def test_allocate_implicit_type(self):
        """Test allocate_implicit_type produces unique virtual signals."""
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)

        allocations = [analyzer.allocate_implicit_type() for _ in range(60)]
        names = [info.name for info in allocations]

        # All allocated names should be unique
        assert len(names) == len(set(names))
        # Implicit signals use __v prefix
        assert names[0] == "__v1"
        assert names[1] == "__v2"
        assert names[59] == "__v60"


class TestSemanticAnalyzerSampleFiles:
    """Tests for semantic analysis on sample files."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    def test_semantic_analysis_sample_files(self, parser):
        """Test semantic analysis on sample files if they exist."""
        # Note: Path updated for new example_programs location
        sample_files = [
            "example_programs/01_basic_arithmetic.facto",
            "example_programs/04_memory.facto",
        ]

        for file_path in sample_files:
            path = Path(file_path)
            if path.exists():
                with open(path) as f:
                    code = f.read()
                program = parser.parse(code)
                diagnostics = ProgramDiagnostics()
                analyzer = SemanticAnalyzer(diagnostics)
                analyzer.visit(program)
                assert not diagnostics.has_errors()


class TestSignalTypeMap:
    """Tests for signal_type_map property."""

    def test_signal_type_map(self):
        """Test signal_type_map returns the registry mappings."""
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        signal_map = analyzer.signal_type_map
        assert isinstance(signal_map, dict)


class TestEmitReservedSignalDiagnostic:
    """Tests for _emit_reserved_signal_diagnostic method."""

    def test_emit_reserved_signal_diagnostic_error(self):
        """Test emitting error for reserved signal-W."""
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        node = NumberLiteral(value=1, line=1, column=1)
        analyzer._emit_reserved_signal_diagnostic("signal-W", node, "in test context")
        assert diagnostics.has_errors()

    def test_emit_reserved_signal_diagnostic_non_reserved(self):
        """Test no error for non-reserved signal."""
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        node = NumberLiteral(value=1, line=1, column=1)
        analyzer._emit_reserved_signal_diagnostic("signal-A", node, "in test context")
        assert not diagnostics.has_errors()


class TestIsVirtualChannel:
    """Tests for _is_virtual_channel static method."""

    def test_is_virtual_channel_none(self):
        """Test _is_virtual_channel with None."""
        assert SemanticAnalyzer._is_virtual_channel(None) is False

    def test_is_virtual_channel_virtual_flag(self):
        """Test _is_virtual_channel with is_virtual=True."""
        signal_info = SignalTypeInfo(name="test", is_virtual=True)
        assert SemanticAnalyzer._is_virtual_channel(signal_info) is True

    def test_is_virtual_channel_signal_prefix(self):
        """Test _is_virtual_channel with signal- prefix."""
        signal_info = SignalTypeInfo(name="signal-A", is_virtual=False)
        assert SemanticAnalyzer._is_virtual_channel(signal_info) is True

    def test_is_virtual_channel_implicit_prefix(self):
        """Test _is_virtual_channel with __ prefix."""
        signal_info = SignalTypeInfo(name="__v1", is_virtual=False)
        assert SemanticAnalyzer._is_virtual_channel(signal_info) is True

    def test_is_virtual_channel_item(self):
        """Test _is_virtual_channel with regular item."""
        signal_info = SignalTypeInfo(name="iron-plate", is_virtual=False)
        assert SemanticAnalyzer._is_virtual_channel(signal_info) is False


class TestResolvePhysicalSignalName:
    """Tests for _resolve_physical_signal_name method."""

    def test_resolve_physical_signal_name_none(self):
        """Test resolving None signal key."""
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        result = analyzer._resolve_physical_signal_name(None)
        assert result is None

    def test_resolve_physical_signal_name_string(self):
        """Test resolving signal name from string mapping."""
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.signal_registry.register("test-signal", "mapped-signal", "virtual")
        result = analyzer._resolve_physical_signal_name("test-signal")
        assert result == "mapped-signal"

    def test_resolve_physical_signal_name_dict(self):
        """Test resolving signal name from dict mapping."""
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.signal_registry.register("test-key", "physical-name", "item")
        result = analyzer._resolve_physical_signal_name("test-key")
        assert result == "physical-name"


class TestLookupSignalCategory:
    """Tests for _lookup_signal_category method."""

    def test_lookup_signal_category_none(self):
        """Test lookup with None."""
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        result = analyzer._lookup_signal_category(None)
        assert result is None

    def test_lookup_signal_category_dict(self):
        """Test lookup from dict mapping."""
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.signal_registry.register("test-sig", "test", "virtual")
        result = analyzer._lookup_signal_category("test-sig")
        assert result == "virtual"


class TestRegisterSignalMetadata:
    """Tests for _register_signal_metadata method."""

    def test_register_signal_metadata_non_signal(self):
        """Test registering metadata for non-signal type."""
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        node = NumberLiteral(value=1, line=1, column=1)
        result = analyzer._register_signal_metadata("x", node, IntValue())
        assert result is None

    def test_register_signal_metadata_signal_value(self):
        """Test registering metadata for signal value."""
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        node = NumberLiteral(value=1, line=1, column=1)
        signal_info = SignalTypeInfo(name="signal-A", is_implicit=False)
        signal_value = SignalValue(signal_type=signal_info)
        result = analyzer._register_signal_metadata("test_sig", node, signal_value, "Signal")
        assert result is not None
        assert result.identifier == "test_sig"
        assert "test_sig" in analyzer.signal_debug_info


class TestGetSignalDebugPayload:
    """Tests for get_signal_debug_payload method."""

    def test_get_signal_debug_payload_existing(self):
        """Test getting debug payload for existing signal."""
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        node = NumberLiteral(value=1, line=1, column=1)
        signal_info = SignalTypeInfo(name="signal-A", is_implicit=False)
        signal_value = SignalValue(signal_type=signal_info)
        analyzer._register_signal_metadata("my_sig", node, signal_value)
        result = analyzer.get_signal_debug_payload("my_sig")
        assert result is not None
        assert result.identifier == "my_sig"

    def test_get_signal_debug_payload_missing(self):
        """Test getting debug payload for non-existent signal."""
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        result = analyzer.get_signal_debug_payload("nonexistent")
        assert result is None


class TestMakeSignalTypeInfo:
    """Tests for make_signal_type_info method."""

    def test_make_signal_type_info_virtual(self):
        """Test making signal type info for virtual signal."""
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        result = analyzer.make_signal_type_info("signal-A")
        assert result.name == "signal-A"
        assert result.is_virtual is True

    def test_make_signal_type_info_implicit(self):
        """Test making signal type info for implicit signal."""
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        result = analyzer.make_signal_type_info("__v1")
        assert result.name == "__v1"
        assert result.is_virtual is True

    def test_make_signal_type_info_item(self):
        """Test making signal type info for item."""
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        result = analyzer.make_signal_type_info("iron-plate")
        assert result.name == "iron-plate"
        assert result.is_virtual is False


class TestValidateSignalTypeWithError:
    """Tests for validate_signal_type_with_error method."""

    def test_validate_signal_type_empty(self):
        """Test validation with empty signal name."""
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        node = NumberLiteral(value=1, line=1, column=1)
        result = analyzer.validate_signal_type_with_error("", node)
        assert result is False

    def test_validate_signal_type_invalid(self):
        """Test validation with invalid signal name."""
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        node = NumberLiteral(value=1, line=1, column=1)
        result = analyzer.validate_signal_type_with_error("invalid signal", node)
        assert result is False
        assert diagnostics.has_errors()


class TestResolveSignalTypeAccess:
    """Tests for resolve_signal_type_access method."""

    def test_resolve_signal_type_access_string(self):
        """Test resolving string type reference."""
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        node = NumberLiteral(value=1, line=1, column=1)
        result = analyzer.resolve_signal_type_access("iron-plate", node)
        assert result == "iron-plate"

    def test_resolve_signal_type_access_invalid_property(self):
        """Test resolving signal type access with invalid property."""
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        node = NumberLiteral(value=1, line=1, column=1)
        type_access = SignalTypeAccess(object_name="sig", property_name="value")
        result = analyzer.resolve_signal_type_access(type_access, node)
        assert result is None
        assert diagnostics.has_errors()

    def test_resolve_signal_type_access_undefined_variable(self):
        """Test resolving signal type access for undefined variable."""
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        node = NumberLiteral(value=1, line=1, column=1)
        type_access = SignalTypeAccess(object_name="undefined_sig", property_name="type")
        result = analyzer.resolve_signal_type_access(type_access, node)
        assert result is None
        assert diagnostics.has_errors()


class TestInferExprType:
    """Tests for infer_expr_type method."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    def test_infer_expr_type_number_literal(self):
        """Test inferring type of number literal."""
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        expr = NumberLiteral(value=42, line=1, column=1)
        result = analyzer.infer_expr_type(expr)
        assert isinstance(result, IntValue)
        assert result.value == 42

    def test_infer_expr_type_read_expr(self, parser):
        """Test inferring type of memory read expression."""
        program = parser.parse('Memory m: "signal-A"; Signal x = m.read();')
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert not diagnostics.has_errors()

    def test_infer_expr_type_undefined_memory_read(self, parser):
        """Test inferring type of undefined memory read."""
        program = parser.parse("Signal x = undefined_mem.read();")
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert diagnostics.has_errors()


class TestTypeNameToSymbolType:
    """Tests for _type_name_to_symbol_type method."""

    def test_type_name_to_symbol_type_entity(self):
        """Test converting Entity type name."""
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        result = analyzer._type_name_to_symbol_type("Entity")
        assert result == SymbolType.ENTITY

    def test_type_name_to_symbol_type_memory(self):
        """Test converting Memory type name."""
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        result = analyzer._type_name_to_symbol_type("Memory")
        assert result == SymbolType.MEMORY


class TestValueMatchesType:
    """Tests for _value_matches_type method."""

    def test_value_matches_type_int(self):
        """Test matching int value type."""
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        result = analyzer._value_matches_type(IntValue(), "Int")
        assert result is True

    def test_value_matches_type_signal(self):
        """Test matching signal value type."""
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        signal_info = SignalTypeInfo(name="signal-A", is_implicit=False)
        result = analyzer._value_matches_type(SignalValue(signal_type=signal_info), "Signal")
        assert result is True


class TestValueTypeName:
    """Tests for _value_type_name method."""

    def test_value_type_name_int(self):
        """Test getting type name for IntValue."""
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        result = analyzer._value_type_name(IntValue())
        assert result == "int"

    def test_value_type_name_signal(self):
        """Test getting type name for SignalValue."""
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        signal_info = SignalTypeInfo(name="signal-A", is_implicit=False)
        result = analyzer._value_type_name(SignalValue(signal_type=signal_info))
        assert result == "Signal"


class TestVisitMemDecl:
    """Tests for visit_MemDecl method."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    def test_visit_mem_decl_explicit_type(self, parser):
        """Test visiting memory declaration with explicit type."""
        program = parser.parse('Memory counter: "signal-A";')
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert not diagnostics.has_errors()
        assert "counter" in analyzer.memory_types


class TestVisitFuncDecl:
    """Tests for visit_FuncDecl method."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    def test_visit_func_decl_simple(self, parser):
        """Test visiting simple function declaration."""
        program = parser.parse("func add(int a, int b) { return a + b; }")
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert not diagnostics.has_errors()

    def test_visit_func_decl_void_return(self, parser):
        """Test visiting function with void return."""
        program = parser.parse('func setup() { Memory m: "signal-A"; }')
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert not diagnostics.has_errors()


class TestVisitForStmt:
    """Tests for visit_ForStmt method."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    def test_visit_for_stmt_range(self, parser):
        """Test visiting for statement with range."""
        program = parser.parse("for i in 0..5 { Signal x = i; }")
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert not diagnostics.has_errors()


class TestVisitReturnStmt:
    """Tests for visit_ReturnStmt method."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    def test_visit_return_stmt(self, parser):
        """Test visiting return statement."""
        program = parser.parse("func test() { return 42; }")
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert not diagnostics.has_errors()


class TestBundleOperations:
    """Tests for bundle-related methods."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    def test_infer_bundle_literal_type(self, parser):
        """Test inferring bundle literal type."""
        program = parser.parse('Bundle b = {("iron-plate", 10), ("copper-plate", 5)};')
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert not diagnostics.has_errors()


class TestWriteExprValidation:
    """Tests for WriteExpr validation."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    def test_write_expr_multiple_writes_error(self, parser):
        """Test that multiple writes to same memory are rejected."""
        code = """
        Memory m: "signal-A";
        m.write(1);
        m.write(2);
        """
        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert diagnostics.has_errors()

    def test_write_expr_latch_invalid_set_type(self, parser):
        """Test latch write with invalid set signal type."""
        code = """
        Memory m: "signal-A";
        m.write(1, set=100, reset=("signal-B", 1));
        """
        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert diagnostics.has_errors()

    def test_write_expr_latch_invalid_reset_type(self, parser):
        """Test latch write with invalid reset signal type."""
        code = """
        Memory m: "signal-A";
        m.write(1, set=("signal-B", 1), reset=200);
        """
        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert diagnostics.has_errors()

    def test_write_expr_latch_valid(self, parser):
        """Test that valid latch write works."""
        code = """
        Memory m: "signal-A";
        m.write(1, set=("signal-B", 1), reset=("signal-C", 1));
        """
        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert not diagnostics.has_errors()


class TestCallExprValidation:
    """Tests for CallExpr validation."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    def test_call_expr_place_validation(self, parser):
        """Test place() call validation."""
        code = 'Entity chest = place("steel-chest", 0, 0, {read_contents: 1});'
        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert not diagnostics.has_errors()

    def test_call_expr_user_function(self, parser):
        """Test user function call."""
        code = """
        func multiply(int a, int b) { return a * b; }
        Signal result = multiply(5, 10);
        """
        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert not diagnostics.has_errors()


class TestBinaryOpTypeInference:
    """Tests for binary operation type inference."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    def test_binary_op_arithmetic_int(self, parser):
        """Test arithmetic operation with integers."""
        program = parser.parse("Signal result = 10 + 20;")
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert not diagnostics.has_errors()

    def test_binary_op_comparison(self, parser):
        """Test comparison operation."""
        program = parser.parse("Signal result = 10 > 5;")
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert not diagnostics.has_errors()

    def test_binary_op_bitwise(self, parser):
        """Test bitwise operation."""
        program = parser.parse("Signal result = 10 AND 5;")
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert not diagnostics.has_errors()


class TestUnaryOpTypeInference:
    """Tests for unary operation type inference."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    def test_unary_op_negation(self, parser):
        """Test negation operation."""
        program = parser.parse("Signal result = -42;")
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert not diagnostics.has_errors()


class TestProjectionExpr:
    """Tests for projection expression."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    def test_projection_expr_simple(self, parser):
        """Test simple projection expression."""
        program = parser.parse('Signal result = 10 | "iron-plate";')
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert not diagnostics.has_errors()


class TestPropertyAccessExpr:
    """Tests for property access expression."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    def test_property_access_entity_output(self, parser):
        """Test property access on entity."""
        code = """
        Entity chest = place("steel-chest", 0, 0, {read_contents: 1});
        Bundle contents = chest.output;
        """
        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert not diagnostics.has_errors()


class TestErrorDetection:
    """Tests for various error detection paths."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    def test_type_access_on_non_signal(self, parser):
        """Test .type access on non-signal variable produces error."""
        code = """
        int x = 5;
        Signal s = x.type: 10;
        """
        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert diagnostics.has_errors()

    def test_multiple_memory_writes(self, parser):
        """Test multiple writes to same memory produces error."""
        code = """
        Memory m: "signal-A";
        m.write(10);
        m.write(20);
        """
        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert diagnostics.has_errors()

    def test_write_to_non_memory(self, parser):
        """Test writing to non-memory variable produces error."""
        code = """
        Signal s = 10;
        s.write(20);
        """
        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert diagnostics.has_errors()

    def test_read_from_non_memory(self, parser):
        """Test reading from non-memory variable produces error."""
        code = """
        Signal s = 10;
        Signal r = s.read();
        """
        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert diagnostics.has_errors()

    def test_undefined_function_call(self, parser):
        """Test calling undefined function produces error."""
        code = """
        Signal r = undefined_func(10);
        """
        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert diagnostics.has_errors()

    def test_function_argument_count_mismatch(self, parser):
        """Test calling function with wrong number of arguments produces error."""
        code = """
        func add(int a, int b) { return a + b; }
        Signal r = add(10);
        """
        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert diagnostics.has_errors()

    def test_undefined_entity_property(self, parser):
        """Test accessing undefined entity property produces error."""
        code = """
        Entity lamp = place("small-lamp", 0, 0, {});
        lamp.undefined_property = 10;
        """
        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert diagnostics.has_errors()

    def test_for_loop_with_invalid_range(self, parser):
        """Test for loop with non-constant range bounds."""
        code = """
        Signal x = 10;
        for i in 0..x { Signal y = i; }
        """
        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        # Non-constant range bounds raise ValueError
        with pytest.raises(ValueError):
            analyzer.visit(program)

    def test_place_with_invalid_prototype(self, parser):
        """Test place() with unknown entity prototype produces warning not error."""
        code = """
        Entity e = place("completely-fake-entity", 0, 0, {enabled: 1});
        """
        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        # Unknown entity produces warning, not error
        assert not diagnostics.has_errors()


class TestTypeInference:
    """Tests for expression type inference."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    def test_infer_number_literal_type(self, parser):
        """Test that number literals are inferred as IntValue."""
        program = parser.parse("Signal x = 42;")
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert not diagnostics.has_errors()

    def test_infer_signal_literal_type(self, parser):
        """Test that signal literals are inferred as SignalValue."""
        program = parser.parse('Signal x = ("iron-plate", 10);')
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert not diagnostics.has_errors()

    def test_infer_bundle_type(self, parser):
        """Test that bundle literals are inferred as BundleValue."""
        program = parser.parse('Bundle b = {("iron-plate", 10), ("copper-plate", 20)};')
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert not diagnostics.has_errors()

    def test_infer_binary_op_type(self, parser):
        """Test that binary operations infer correct type."""
        program = parser.parse("Signal x = 10 + 20;")
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert not diagnostics.has_errors()

    def test_infer_comparison_type(self, parser):
        """Test that comparison operations infer boolean type."""
        program = parser.parse("Signal x = 10 > 5;")
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert not diagnostics.has_errors()

    def test_infer_unary_op_type(self, parser):
        """Test that unary operations infer correct type."""
        program = parser.parse("Signal x = -42;")
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert not diagnostics.has_errors()


class TestFunctionAnalysis:
    """Tests for function declaration analysis."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    def test_function_with_parameters(self, parser):
        """Test function with parameters is analyzed correctly."""
        code = """
        func add(int a, int b) { return a + b; }
        Signal result = add(10, 20);
        """
        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert not diagnostics.has_errors()

    def test_function_return_type(self, parser):
        """Test function return type is inferred."""
        code = """
        func double(int x) { return x * 2; }
        Signal result = double(21);
        """
        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert not diagnostics.has_errors()


class TestEntityAnalysis:
    """Tests for entity declaration analysis."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    def test_entity_with_properties(self, parser):
        """Test entity with properties is analyzed correctly."""
        code = """
        Entity lamp = place("small-lamp", 0, 0, {enabled: 1, use_colors: 1});
        """
        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert not diagnostics.has_errors()

    def test_entity_property_assignment(self, parser):
        """Test entity property assignment is analyzed correctly."""
        code = """
        Entity lamp = place("small-lamp", 0, 0, {enabled: 1, use_colors: 1});
        lamp.enable = 10 > 5;
        """
        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert not diagnostics.has_errors()

    def test_entity_color_properties(self, parser):
        """Test entity color property assignment."""
        code = """
        Entity lamp = place("small-lamp", 0, 0, {use_colors: 1});
        lamp.color_r = 255;
        lamp.color_g = 128;
        lamp.color_b = 64;
        """
        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert not diagnostics.has_errors()


class TestBundleAnalysis:
    """Tests for bundle expression analysis."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    def test_bundle_selection(self, parser):
        """Test bundle selection expression."""
        code = """
        Bundle b = {("iron-plate", 10), ("copper-plate", 20)};
        Signal iron = b["iron-plate"];
        """
        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert not diagnostics.has_errors()

    def test_bundle_all_expr(self, parser):
        """Test all(bundle) expression."""
        code = """
        Bundle b = {("iron-plate", 10), ("copper-plate", 20)};
        Signal total = all(b);
        """
        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert not diagnostics.has_errors()

    def test_bundle_any_expr(self, parser):
        """Test any(bundle) expression."""
        code = """
        Bundle b = {("iron-plate", 0), ("copper-plate", 20)};
        Signal has_any = any(b);
        """
        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert not diagnostics.has_errors()


class TestProjectionAnalysis:
    """Tests for projection expression analysis."""

    @pytest.fixture
    def parser(self):
        return DSLParser()

    def test_projection_to_item_signal(self, parser):
        """Test projection to item signal."""
        code = """
        Signal x = 10 | "iron-plate";
        """
        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert not diagnostics.has_errors()

    def test_projection_to_virtual_signal(self, parser):
        """Test projection to virtual signal."""
        code = """
        Signal x = 10 | "signal-A";
        """
        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert not diagnostics.has_errors()

    def test_projection_chain(self, parser):
        """Test chained projections."""
        code = """
        Signal x = 10 | "iron-plate";
        Signal y = x | "copper-plate";
        """
        program = parser.parse(code)
        diagnostics = ProgramDiagnostics()
        analyzer = SemanticAnalyzer(diagnostics)
        analyzer.visit(program)
        assert not diagnostics.has_errors()
