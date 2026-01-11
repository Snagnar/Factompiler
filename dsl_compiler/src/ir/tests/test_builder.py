"""
Tests for ir/builder.py - IR builder functionality.
"""

from dsl_compiler.src.ir.builder import IRBuilder, SignalRef
from dsl_compiler.src.ir.nodes import IRArith, IRConst, IRDecider


class TestIRBuilder:
    """Tests for IRBuilder class."""

    def test_ir_builder_initialization(self):
        """Test IRBuilder can be initialized."""
        builder = IRBuilder()
        assert builder is not None

    def test_ir_builder_has_signal_type_map(self):
        """Test IRBuilder has a signal_type_map."""
        builder = IRBuilder()
        assert hasattr(builder, "signal_type_map")
        assert isinstance(builder.signal_type_map, dict)

    def test_next_id_generates_unique_ids(self):
        """Test next_id generates unique identifiers."""
        builder = IRBuilder()
        ids = [builder.next_id("test") for _ in range(5)]
        assert len(ids) == len(set(ids))
        assert ids[0] == "test_1"
        assert ids[4] == "test_5"

    def test_add_operation_registers_operation(self):
        """Test add_operation registers the operation for lookup."""
        builder = IRBuilder()
        const_ref = builder.const("signal-A", 42)
        op = builder.get_operation(const_ref.source_id)
        assert op is not None
        assert isinstance(op, IRConst)

    def test_get_operation_returns_none_for_unknown(self):
        """Test get_operation returns None for unknown IDs."""
        builder = IRBuilder()
        assert builder.get_operation("nonexistent_id") is None


class TestIRBuilderConst:
    """Tests for const method."""

    def test_const_creates_ir_const(self):
        """Test const creates an IRConst operation."""
        builder = IRBuilder()
        ref = builder.const("signal-A", 100)
        assert isinstance(ref, SignalRef)
        op = builder.get_operation(ref.source_id)
        assert isinstance(op, IRConst)
        assert op.value == 100


class TestIRBuilderArithmetic:
    """Tests for arithmetic method."""

    def test_arithmetic_creates_ir_arith(self):
        """Test arithmetic creates an IRArith operation."""
        builder = IRBuilder()
        left = builder.const("signal-A", 10)
        right = builder.const("signal-B", 20)
        result = builder.arithmetic("+", left, right, "signal-C")
        assert isinstance(result, SignalRef)
        op = builder.get_operation(result.source_id)
        assert isinstance(op, IRArith)
        assert op.op == "+"


class TestIRBuilderDecider:
    """Tests for decider method."""

    def test_decider_creates_ir_decider(self):
        """Test decider creates an IRDecider operation."""
        builder = IRBuilder()
        left = builder.const("signal-A", 10)
        result = builder.decider(">", left, 5, 1, "signal-B")
        assert isinstance(result, SignalRef)
        op = builder.get_operation(result.source_id)
        assert isinstance(op, IRDecider)
        assert op.test_op == ">"

    def test_decider_with_copy_count(self):
        """Test decider with copy_count_from_input."""
        builder = IRBuilder()
        left = builder.const("signal-A", 10)
        result = builder.decider(">", left, 5, left, "signal-B", copy_count_from_input=True)
        op = builder.get_operation(result.source_id)
        assert op.copy_count_from_input is True


class TestIRBuilderDeciderMulti:
    """Tests for decider_multi method."""

    def test_decider_multi_creates_multi_condition(self):
        """Test decider_multi creates a multi-condition decider."""
        builder = IRBuilder()
        sig_a = builder.const("signal-A", 10)
        sig_b = builder.const("signal-B", 20)
        conditions = [
            (">", sig_a, 5),
            ("<", sig_b, 30),
        ]
        result = builder.decider_multi(conditions, "and", 1, "signal-C")
        op = builder.get_operation(result.source_id)
        assert isinstance(op, IRDecider)
        assert len(op.conditions) == 2


class TestIRBuilderImplicitSignalAllocation:
    """Tests for implicit signal allocation in IRBuilder."""

    def test_allocate_implicit_type(self):
        """Test allocate_implicit_type produces unique names."""
        builder = IRBuilder()
        names = [builder.allocate_implicit_type() for _ in range(60)]

        # All allocated names should be unique
        assert len(names) == len(set(names))

    def test_allocate_implicit_type_prefix(self):
        """Test implicit signals use __v prefix with sequential numbering."""
        builder = IRBuilder()
        names = [builder.allocate_implicit_type() for _ in range(5)]

        assert names[0] == "__v1"
        assert names[1] == "__v2"
        assert names[2] == "__v3"
        assert names[3] == "__v4"
        assert names[4] == "__v5"

    def test_allocate_more_than_26_virtual_signals(self):
        """Ensure implicit signal allocation scales beyond 26 signals."""
        builder = IRBuilder()
        names = [builder.allocate_implicit_type() for _ in range(60)]

        # All allocated names should be unique
        assert len(names) == len(set(names))
        # Implicit signals use __v prefix with sequential numbering
        assert names[0] == "__v1"
        assert names[59] == "__v60"


class TestIRBuilderWireMerge:
    """Tests for wire_merge method."""

    def test_wire_merge_creates_merge_node(self):
        """Test wire_merge creates an IRWireMerge operation."""
        from dsl_compiler.src.ir.nodes import IRWireMerge

        builder = IRBuilder()
        src1 = builder.const("signal-A", 10)
        src2 = builder.const("signal-A", 20)
        result = builder.wire_merge([src1, src2], "signal-A")
        assert isinstance(result, SignalRef)
        op = builder.get_operation(result.source_id)
        assert isinstance(op, IRWireMerge)


class TestIRBuilderMemory:
    """Tests for memory operations."""

    def test_memory_create_registers_operation(self):
        """Test memory_create registers an IRMemCreate."""
        from dsl_compiler.src.ir.nodes import IRMemCreate

        builder = IRBuilder()
        builder.memory_create("mem1", "signal-A")
        # Memory create uses memory_id as the operation ID
        ops = [op for op in builder.get_ir() if isinstance(op, IRMemCreate)]
        assert len(ops) == 1
        assert ops[0].memory_id == "mem1"

    def test_memory_read_returns_signal_ref(self):
        """Test memory_read returns a SignalRef."""
        builder = IRBuilder()
        builder.memory_create("mem1", "signal-A")
        ref = builder.memory_read("mem1", "signal-A")
        assert isinstance(ref, SignalRef)
        assert ref.signal_type == "signal-A"

    def test_memory_write_returns_operation(self):
        """Test memory_write returns an IRMemWrite."""
        from dsl_compiler.src.ir.nodes import IRMemWrite

        builder = IRBuilder()
        builder.memory_create("mem1", "signal-A")
        data = builder.const("signal-A", 42)
        enable = builder.const("signal-B", 1)
        op = builder.memory_write("mem1", data, enable)
        assert isinstance(op, IRMemWrite)

    def test_latch_write_returns_operation(self):
        """Test latch_write returns an IRLatchWrite."""
        from dsl_compiler.src.ir.nodes import MEMORY_TYPE_SR_LATCH, IRLatchWrite

        builder = IRBuilder()
        builder.memory_create("latch1", "signal-A", memory_type=MEMORY_TYPE_SR_LATCH)
        value = builder.const("signal-A", 1)
        set_sig = builder.const("signal-B", 1)
        reset_sig = builder.const("signal-C", 1)
        op = builder.latch_write("latch1", value, set_sig, reset_sig, MEMORY_TYPE_SR_LATCH)
        assert isinstance(op, IRLatchWrite)


class TestIRBuilderEntity:
    """Tests for entity placement."""

    def test_place_entity_registers_operation(self):
        """Test place_entity registers an IRPlaceEntity."""
        from dsl_compiler.src.ir.nodes import IRPlaceEntity

        builder = IRBuilder()
        builder.place_entity("ent1", "lamp", 0, 0, {"enabled": True})
        ops = [op for op in builder.get_ir() if isinstance(op, IRPlaceEntity)]
        assert len(ops) == 1
        assert ops[0].prototype == "lamp"


class TestIRBuilderBundle:
    """Tests for bundle operations."""

    def test_bundle_const_creates_bundle_ref(self):
        """Test bundle_const creates a BundleRef."""
        from dsl_compiler.src.ir.builder import BundleRef

        builder = IRBuilder()
        ref = builder.bundle_const({"signal-A": 10, "signal-B": 20})
        assert isinstance(ref, BundleRef)
        assert "signal-A" in ref.signal_types
        assert "signal-B" in ref.signal_types

    def test_bundle_arithmetic_returns_bundle_ref(self):
        """Test bundle_arithmetic returns a BundleRef with same signals."""
        from dsl_compiler.src.ir.builder import BundleRef

        builder = IRBuilder()
        bundle = builder.bundle_const({"signal-A": 10, "signal-B": 20})
        result = builder.bundle_arithmetic("+", bundle, 5)
        assert isinstance(result, BundleRef)
        assert result.signal_types == bundle.signal_types

    def test_bundle_arithmetic_with_signal_sets_wire_separation(self):
        """Test bundle arithmetic with signal operand sets wire separation."""
        from dsl_compiler.src.ir.nodes import IRArith

        builder = IRBuilder()
        bundle = builder.bundle_const({"signal-A": 10})
        scalar = builder.const("signal-C", 5)
        result = builder.bundle_arithmetic("*", bundle, scalar)
        op = builder.get_operation(result.source_id)
        assert isinstance(op, IRArith)
        assert op.needs_wire_separation is True


class TestIRBuilderGetIR:
    """Tests for get_ir method."""

    def test_get_ir_returns_copy(self):
        """Test get_ir returns a copy of operations."""
        builder = IRBuilder()
        builder.const("signal-A", 10)
        ops1 = builder.get_ir()
        ops2 = builder.get_ir()
        assert ops1 is not ops2
        assert len(ops1) == len(ops2)
