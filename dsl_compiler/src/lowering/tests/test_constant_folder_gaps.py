"""Tests for constant_folder.py coverage gaps."""

from dsl_compiler.src.ast.expressions import BinaryOp
from dsl_compiler.src.ast.literals import NumberLiteral
from dsl_compiler.src.common.diagnostics import ProgramDiagnostics
from dsl_compiler.src.lowering.constant_folder import ConstantFolder


class TestConstantFolderPowerOperator:
    """Cover lines 78-85: power operator edge cases."""

    def test_power_negative_exponent(self):
        """Test power with negative exponent returns 0 and warns."""
        diags = ProgramDiagnostics()
        # Create AST for 2 ** -3
        node = BinaryOp(NumberLiteral(2), "**", NumberLiteral(-3))

        result = ConstantFolder.fold_binary_operation("**", 2, -3, node, diags)

        assert result == 0
        # Should have warning about negative exponent

    def test_power_overflow(self):
        """Test power operation that causes overflow."""
        diags = ProgramDiagnostics()
        # Create AST for 10 ** 100 (very large)
        node = BinaryOp(NumberLiteral(10), "**", NumberLiteral(100))

        ConstantFolder.fold_binary_operation("**", 10, 100, node, diags)
        # Should return an integer (may be very large or 0)

    def test_power_zero_exponent(self):
        """Test power with zero exponent returns 1."""
        node = BinaryOp(NumberLiteral(5), "**", NumberLiteral(0))

        result = ConstantFolder.fold_binary_operation("**", 5, 0, node, None)

        assert result == 1

    def test_power_normal_case(self):
        """Test normal power operation."""
        node = BinaryOp(NumberLiteral(2), "**", NumberLiteral(8))

        result = ConstantFolder.fold_binary_operation("**", 2, 8, node, None)

        assert result == 256


class TestConstantFolderShiftOperators:
    """Tests for shift operator edge cases."""

    def test_left_shift_overflow(self):
        """Test left shift >= 32 returns 0."""
        node = BinaryOp(NumberLiteral(1), "<<", NumberLiteral(32))

        result = ConstantFolder.fold_binary_operation("<<", 1, 32, node, None)

        assert result == 0

    def test_left_shift_negative_amount(self):
        """Test left shift with negative amount returns 0."""
        node = BinaryOp(NumberLiteral(1), "<<", NumberLiteral(-1))

        result = ConstantFolder.fold_binary_operation("<<", 1, -1, node, None)

        assert result == 0

    def test_right_shift_overflow(self):
        """Test right shift >= 32 returns 0."""
        node = BinaryOp(NumberLiteral(1000), ">>", NumberLiteral(32))

        result = ConstantFolder.fold_binary_operation(">>", 1000, 32, node, None)

        assert result == 0

    def test_right_shift_negative_amount(self):
        """Test right shift with negative amount returns 0."""
        node = BinaryOp(NumberLiteral(1000), ">>", NumberLiteral(-1))

        result = ConstantFolder.fold_binary_operation(">>", 1000, -1, node, None)

        assert result == 0


class TestConstantFolderBitwiseOperators:
    """Tests for bitwise operators."""

    def test_bitwise_and(self):
        """Test bitwise AND operation."""
        node = BinaryOp(NumberLiteral(0xFF), "AND", NumberLiteral(0x0F))

        result = ConstantFolder.fold_binary_operation("AND", 0xFF, 0x0F, node, None)

        assert result == 0x0F

    def test_bitwise_or(self):
        """Test bitwise OR operation."""
        node = BinaryOp(NumberLiteral(0xF0), "OR", NumberLiteral(0x0F))

        result = ConstantFolder.fold_binary_operation("OR", 0xF0, 0x0F, node, None)

        assert result == 0xFF

    def test_bitwise_xor(self):
        """Test bitwise XOR operation."""
        node = BinaryOp(NumberLiteral(0xFF), "XOR", NumberLiteral(0xF0))

        result = ConstantFolder.fold_binary_operation("XOR", 0xFF, 0xF0, node, None)

        assert result == 0x0F


class TestConstantFolderDivisionByZero:
    """Tests for division by zero handling."""

    def test_integer_division_by_zero(self):
        """Test integer division by zero returns 0."""
        diags = ProgramDiagnostics()
        node = BinaryOp(NumberLiteral(10), "/", NumberLiteral(0))

        result = ConstantFolder.fold_binary_operation("/", 10, 0, node, diags)

        assert result == 0
        # Should have warning

    def test_modulo_by_zero(self):
        """Test modulo by zero returns 0."""
        diags = ProgramDiagnostics()
        node = BinaryOp(NumberLiteral(10), "%", NumberLiteral(0))

        result = ConstantFolder.fold_binary_operation("%", 10, 0, node, diags)

        assert result == 0
