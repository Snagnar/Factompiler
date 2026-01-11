"""
Tests for semantic/type_system.py - Type system primitives.
"""

from dsl_compiler.src.ast.literals import NumberLiteral
from dsl_compiler.src.common.signal_registry import SignalTypeInfo
from dsl_compiler.src.semantic.type_system import (
    BundleValue,
    DynamicBundleValue,
    EntityValue,
    FunctionValue,
    IntValue,
    MemoryInfo,
    SignalDebugInfo,
    SignalValue,
    VoidValue,
    get_signal_type_name,
)


class TestSignalValue:
    """Tests for SignalValue dataclass."""

    def test_signal_value_creation(self):
        """Test creating a SignalValue."""
        signal_type = SignalTypeInfo(name="iron-plate", is_implicit=False)
        signal = SignalValue(signal_type=signal_type)
        assert signal.signal_type == signal_type
        assert signal.count_expr is None

    def test_signal_value_with_count_expr(self):
        """Test SignalValue with count expression."""
        signal_type = SignalTypeInfo(name="copper-plate", is_implicit=False)
        count_expr = NumberLiteral(value=5, line=1, column=1)
        signal = SignalValue(signal_type=signal_type, count_expr=count_expr)
        assert signal.count_expr == count_expr


class TestSignalDebugInfo:
    """Tests for SignalDebugInfo dataclass."""

    def test_signal_debug_info_creation(self):
        """Test creating SignalDebugInfo."""
        node = NumberLiteral(value=1, line=1, column=1)
        debug_info = SignalDebugInfo(
            identifier="test_signal",
            signal_key="signal-A",
            factorio_signal="signal-A",
            source_node=node,
        )
        assert debug_info.identifier == "test_signal"
        assert debug_info.signal_key == "signal-A"
        assert debug_info.factorio_signal == "signal-A"
        assert debug_info.source_node == node

    def test_signal_debug_info_as_dict(self):
        """Test SignalDebugInfo.as_dict() method."""
        node = NumberLiteral(value=1, line=1, column=1)
        debug_info = SignalDebugInfo(
            identifier="my_signal",
            signal_key="signal-B",
            factorio_signal="signal-B",
            source_node=node,
            declared_type="Signal",
            location="test.facto:1:1",
            category="virtual",
        )
        result = debug_info.as_dict()
        assert result["name"] == "my_signal"
        assert result["signal_key"] == "signal-B"
        assert result["factorio_signal"] == "signal-B"
        assert result["declared_type"] == "Signal"
        assert result["location"] == "test.facto:1:1"
        assert result["category"] == "virtual"
        assert result["source_ast"] == node


class TestIntValue:
    """Tests for IntValue dataclass."""

    def test_int_value_without_value(self):
        """Test IntValue without value (computed)."""
        int_val = IntValue()
        assert int_val.value is None

    def test_int_value_with_value(self):
        """Test IntValue with constant value."""
        int_val = IntValue(value=42)
        assert int_val.value == 42


class TestFunctionValue:
    """Tests for FunctionValue dataclass."""

    def test_function_value_defaults(self):
        """Test FunctionValue with defaults."""
        func_val = FunctionValue()
        assert func_val.param_types == []
        assert isinstance(func_val.return_type, IntValue)

    def test_function_value_with_params(self):
        """Test FunctionValue with parameters."""
        param1 = IntValue()
        param2 = SignalValue(signal_type=SignalTypeInfo(name="iron-plate", is_implicit=False))
        func_val = FunctionValue(param_types=[param1, param2], return_type=VoidValue())
        assert len(func_val.param_types) == 2
        assert isinstance(func_val.return_type, VoidValue)


class TestEntityValue:
    """Tests for EntityValue dataclass."""

    def test_entity_value_creation(self):
        """Test creating EntityValue."""
        entity = EntityValue(entity_id="chest_1", prototype="steel-chest")
        assert entity.entity_id == "chest_1"
        assert entity.prototype == "steel-chest"

    def test_entity_value_defaults(self):
        """Test EntityValue defaults."""
        entity = EntityValue()
        assert entity.entity_id is None
        assert entity.prototype is None


class TestVoidValue:
    """Tests for VoidValue dataclass."""

    def test_void_value_creation(self):
        """Test creating VoidValue."""
        void = VoidValue()
        assert void is not None


class TestBundleValue:
    """Tests for BundleValue dataclass."""

    def test_bundle_value_empty(self):
        """Test empty BundleValue."""
        bundle = BundleValue()
        assert bundle.signal_types == set()

    def test_bundle_value_with_signal_types(self):
        """Test BundleValue with signal types."""
        signal_types = {"iron-plate", "copper-plate", "signal-A"}
        bundle = BundleValue(signal_types=signal_types)
        assert bundle.signal_types == signal_types


class TestDynamicBundleValue:
    """Tests for DynamicBundleValue dataclass."""

    def test_dynamic_bundle_value_defaults(self):
        """Test DynamicBundleValue defaults."""
        bundle = DynamicBundleValue()
        assert bundle.source_entity_id == ""
        assert bundle.is_dynamic is True
        assert bundle.signal_types == set()

    def test_dynamic_bundle_value_with_entity(self):
        """Test DynamicBundleValue with entity ID."""
        bundle = DynamicBundleValue(source_entity_id="chest_1")
        assert bundle.source_entity_id == "chest_1"
        assert bundle.is_dynamic is True


class TestMemoryInfo:
    """Tests for MemoryInfo dataclass."""

    def test_memory_info_creation(self):
        """Test creating MemoryInfo."""
        signal_info = SignalTypeInfo(name="signal-X", is_implicit=False)
        mem_info = MemoryInfo(
            name="counter",
            symbol=None,
            signal_type="signal-X",
            signal_info=signal_info,
            explicit=True,
        )
        assert mem_info.name == "counter"
        assert mem_info.signal_type == "signal-X"
        assert mem_info.signal_info == signal_info
        assert mem_info.explicit is True

    def test_memory_info_defaults(self):
        """Test MemoryInfo defaults."""
        mem_info = MemoryInfo(name="mem", symbol=None)
        assert mem_info.signal_type is None
        assert mem_info.signal_info is None
        assert mem_info.explicit is False


class TestGetSignalTypeName:
    """Tests for get_signal_type_name function."""

    def test_get_signal_type_name_from_signal_value(self):
        """Test extracting signal type name from SignalValue."""
        signal_type = SignalTypeInfo(name="iron-plate", is_implicit=False)
        signal_value = SignalValue(signal_type=signal_type)
        result = get_signal_type_name(signal_value)
        assert result == "iron-plate"

    def test_get_signal_type_name_from_int_value(self):
        """Test get_signal_type_name returns None for IntValue."""
        int_value = IntValue(value=42)
        result = get_signal_type_name(int_value)
        assert result is None

    def test_get_signal_type_name_from_bundle_value(self):
        """Test get_signal_type_name returns None for BundleValue."""
        bundle = BundleValue(signal_types={"iron-plate"})
        result = get_signal_type_name(bundle)
        assert result is None

    def test_get_signal_type_name_from_entity_value(self):
        """Test get_signal_type_name returns None for EntityValue."""
        entity = EntityValue(entity_id="chest_1")
        result = get_signal_type_name(entity)
        assert result is None
