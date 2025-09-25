# test_parser.py
"""
Comprehensive test suite for the DSL parser.
"""

import pytest
import sys
from pathlib import Path

from dsl_compiler.src.parser import DSLParser
from dsl_compiler.src.dsl_ast import *


class TestDSLParser:
    """Comprehensive DSL parser tests."""
    
    def setup_method(self):
        """Set up parser for each test."""
        self.parser = DSLParser()
    
    def test_literals(self):
        """Test parsing of literals."""
        test_cases = [
            ("let x = 42;", NumberLiteral, 42),
            ('let s = "hello";', StringLiteral, "hello"),
            ("let neg = -17;", NumberLiteral, -17),  # negative number parsed directly
        ]
        
        for code, expected_type, expected_value in test_cases:
            ast = self.parser.parse(code)
            assert len(ast.statements) == 1
            stmt = ast.statements[0]
            assert isinstance(stmt, LetStmt)
            assert isinstance(stmt.value, expected_type)
            if expected_value is not None:
                assert stmt.value.value == expected_value
    
    def test_identifiers_and_property_access(self):
        """Test parsing identifiers and property access."""
        # Simple identifier
        ast = self.parser.parse("let x = y;")
        stmt = ast.statements[0]
        assert isinstance(stmt.value, IdentifierExpr)
        assert stmt.value.name == "y"
        
        # Property access in lvalue
        ast = self.parser.parse("lamp.enable = true;")
        stmt = ast.statements[0]
        assert isinstance(stmt, AssignStmt)
        assert isinstance(stmt.target, PropertyAccess)
        assert stmt.target.object_name == "lamp"
        assert stmt.target.property_name == "enable"
    
    def test_arithmetic_expressions(self):
        """Test parsing arithmetic expressions with correct precedence."""
        test_cases = [
            ("let x = a + b;", BinaryOp, "+"),
            ("let x = a - b;", BinaryOp, "-"),
            ("let x = a * b;", BinaryOp, "*"),
            ("let x = a / b;", BinaryOp, "/"),
            ("let x = a % b;", BinaryOp, "%"),
            ("let x = a + b * c;", BinaryOp, "+"),  # Should be a + (b * c)
        ]
        
        for code, expected_type, expected_op in test_cases:
            ast = self.parser.parse(code)
            stmt = ast.statements[0]
            assert isinstance(stmt.value, expected_type)
            assert stmt.value.op == expected_op
            
        # Test precedence: a + b * c should parse as a + (b * c)
        ast = self.parser.parse("let x = a + b * c;")
        stmt = ast.statements[0]
        assert isinstance(stmt.value, BinaryOp)
        assert stmt.value.op == "+"
        assert isinstance(stmt.value.right, BinaryOp)
        assert stmt.value.right.op == "*"
    
    def test_comparison_expressions(self):
        """Test parsing comparison expressions."""
        test_cases = ["==", "!=", "<", ">", "<=", ">="]
        
        for op in test_cases:
            code = f"let x = a {op} b;"
            ast = self.parser.parse(code)
            stmt = ast.statements[0]
            assert isinstance(stmt.value, BinaryOp)
            assert stmt.value.op == op
    
    def test_logical_expressions(self):
        """Test parsing logical expressions."""
        test_cases = ["&&", "||"]
        
        for op in test_cases:
            code = f"let x = a {op} b;"
            ast = self.parser.parse(code)
            stmt = ast.statements[0]
            assert isinstance(stmt.value, BinaryOp)
            assert stmt.value.op == op
    
    def test_unary_expressions(self):
        """Test parsing unary expressions."""
        test_cases = ["+", "-", "!"]
        
        for op in test_cases:
            code = f"let x = {op}a;"
            ast = self.parser.parse(code)
            stmt = ast.statements[0]
            assert isinstance(stmt.value, UnaryOp)
            assert stmt.value.op == op
    
    def test_projection_expressions(self):
        """Test parsing projection expressions (| operator)."""
        ast = self.parser.parse('let x = a | "signal-type";')
        stmt = ast.statements[0]
        assert isinstance(stmt.value, ProjectionExpr)
        assert stmt.value.target_type == "signal-type"
        
        # Test chained projections
        ast = self.parser.parse('let x = a | "type1" | "type2";')
        stmt = ast.statements[0]
        assert isinstance(stmt.value, ProjectionExpr)
        assert stmt.value.target_type == "type2"
        assert isinstance(stmt.value.expr, ProjectionExpr)
        assert stmt.value.expr.target_type == "type1"
    
    def test_input_expressions(self):
        """Test parsing input expressions."""
        # Simple input by index
        ast = self.parser.parse("let x = input(0);")
        stmt = ast.statements[0]
        assert isinstance(stmt.value, InputExpr)
        assert isinstance(stmt.value.index, NumberLiteral)
        assert stmt.value.index.value == 0
        assert stmt.value.signal_type is None
        
        # Typed input
        ast = self.parser.parse('let x = input("iron-plate", 1);')
        stmt = ast.statements[0]
        assert isinstance(stmt.value, InputExpr)
        assert isinstance(stmt.value.index, NumberLiteral)
        assert stmt.value.index.value == 1
        assert isinstance(stmt.value.signal_type, StringLiteral)
        assert stmt.value.signal_type.value == "iron-plate"
    
    def test_memory_operations(self):
        """Test parsing memory operations."""
        # Memory declaration
        ast = self.parser.parse("mem counter = memory();")
        stmt = ast.statements[0]
        assert isinstance(stmt, MemDecl)
        assert stmt.name == "counter"
        assert stmt.init_expr is None
        
        # Memory declaration with initial value
        ast = self.parser.parse("mem counter = memory(42);")
        stmt = ast.statements[0]
        assert isinstance(stmt, MemDecl)
        assert stmt.name == "counter"
        assert isinstance(stmt.init_expr, NumberLiteral)
        assert stmt.init_expr.value == 42
        
        # Read operation
        ast = self.parser.parse("let x = read(counter);")
        stmt = ast.statements[0]
        assert isinstance(stmt.value, ReadExpr)
        assert stmt.value.memory_name == "counter"
        
        # Write operation
        ast = self.parser.parse("write(counter, x + 1);")
        stmt = ast.statements[0]
        assert isinstance(stmt, ExprStmt)
        assert isinstance(stmt.expr, WriteExpr)
        assert stmt.expr.memory_name == "counter"
        assert isinstance(stmt.expr.value, BinaryOp)
    
    def test_bundle_expressions(self):
        """Test parsing bundle expressions."""
        # Empty bundle
        ast = self.parser.parse("let empty = bundle();")
        stmt = ast.statements[0]
        assert isinstance(stmt.value, BundleExpr)
        assert len(stmt.value.exprs) == 0
        
        # Bundle with arguments
        ast = self.parser.parse("let resources = bundle(iron, copper, coal);")
        stmt = ast.statements[0]
        assert isinstance(stmt.value, BundleExpr)
        assert len(stmt.value.exprs) == 3
        assert all(isinstance(expr, IdentifierExpr) for expr in stmt.value.exprs)
    
    def test_function_calls(self):
        """Test parsing function calls."""
        # Simple call
        ast = self.parser.parse("let x = func();")
        stmt = ast.statements[0]
        assert isinstance(stmt.value, CallExpr)
        assert stmt.value.name == "func"
        assert len(stmt.value.args) == 0
        
        # Call with arguments
        ast = self.parser.parse("let x = func(a, b, 42);")
        stmt = ast.statements[0]
        assert isinstance(stmt.value, CallExpr)
        assert stmt.value.name == "func"
        assert len(stmt.value.args) == 3
        
        # Method call
        ast = self.parser.parse("let x = module.method(arg);")
        stmt = ast.statements[0]
        assert isinstance(stmt.value, CallExpr)
        assert stmt.value.name == "module.method"
        assert len(stmt.value.args) == 1
    
    def test_function_declarations(self):
        """Test parsing function declarations."""
        # Function with no parameters
        code = """
        func simple() {
            return 42;
        }
        """
        ast = self.parser.parse(code)
        stmt = ast.statements[0]
        assert isinstance(stmt, FuncDecl)
        assert stmt.name == "simple"
        assert len(stmt.params) == 0
        assert len(stmt.body) == 1
        assert isinstance(stmt.body[0], ReturnStmt)
        
        # Function with parameters
        code = """
        func add(a, b) {
            let sum = a + b;
            return sum;
        }
        """
        ast = self.parser.parse(code)
        stmt = ast.statements[0]
        assert isinstance(stmt, FuncDecl)
        assert stmt.name == "add"
        assert stmt.params == ["a", "b"]
        assert len(stmt.body) == 2
    
    def test_statements(self):
        """Test parsing different statement types."""
        # Let statement
        ast = self.parser.parse("let x = 42;")
        assert isinstance(ast.statements[0], LetStmt)
        
        # Assignment statement
        ast = self.parser.parse("x = 42;")
        assert isinstance(ast.statements[0], AssignStmt)
        
        # Expression statement
        ast = self.parser.parse("myFunc();")
        assert isinstance(ast.statements[0], ExprStmt)
        
        # Return statement
        ast = self.parser.parse("return x;")
        assert isinstance(ast.statements[0], ReturnStmt)
        
        # Import statement
        ast = self.parser.parse('import "module";')
        stmt = ast.statements[0]
        assert isinstance(stmt, ImportStmt)
        assert stmt.path == "module"
        assert stmt.alias is None
        
        # Import with alias
        ast = self.parser.parse('import "module" as m;')
        stmt = ast.statements[0]
        assert isinstance(stmt, ImportStmt)
        assert stmt.path == "module"
        assert stmt.alias == "m"
    
    def test_parenthesized_expressions(self):
        """Test parsing parenthesized expressions."""
        ast = self.parser.parse("let x = (a + b) * c;")
        stmt = ast.statements[0]
        assert isinstance(stmt.value, BinaryOp)
        assert stmt.value.op == "*"
        assert isinstance(stmt.value.left, BinaryOp)
        assert stmt.value.left.op == "+"
    
    def test_complex_expressions(self):
        """Test parsing complex nested expressions."""
        code = """
        let result = bundle(
            input("iron-plate", 0) + input("copper-plate", 1),
            read(memory) * 2
        ) | "output-signal";
        """
        ast = self.parser.parse(code)
        stmt = ast.statements[0]
        assert isinstance(stmt, LetStmt)
        assert isinstance(stmt.value, ProjectionExpr)
        assert isinstance(stmt.value.expr, BundleExpr)
        assert len(stmt.value.expr.exprs) == 2
    
    def test_all_sample_files(self):
        """Test parsing all sample .fcdsl files."""
        test_dir = Path(__file__).parent.parent / "tests" / "sample_programs"
        if not test_dir.exists():
            pytest.skip("Sample files directory not found")
        
        test_files = list(test_dir.glob("*.fcdsl"))
        assert len(test_files) > 0, "No test files found"
        
        for test_file in test_files:
            try:
                ast = self.parser.parse_file(test_file)
                assert isinstance(ast, Program)
                assert len(ast.statements) > 0
            except Exception as e:
                pytest.fail(f"Failed to parse {test_file.name}: {e}")
    
    def test_syntax_errors(self):
        """Test that syntax errors are properly detected."""
        invalid_cases = [
            "let x =;",  # Missing expression
            "let = 42;",  # Missing variable name
            "x + ;",  # Incomplete expression
            "func() { return; }",  # Missing return value
            "input();",  # Missing input arguments
        ]
        
        for code in invalid_cases:
            with pytest.raises((SyntaxError, RuntimeError)):
                self.parser.parse(code)


def test_basic_parsing():
    """Legacy test function for backward compatibility."""
    parser = DSLParser()
    
    test_cases = [
        # Basic arithmetic
        """
        let a = input(0);
        let b = input("iron-plate", 1);
        let sum = a + b;
        """,
        
        # Memory operations
        """
        mem counter = memory(0);
        let current = read(counter);
        write(counter, current + 1);
        """,
        
        # Projection and bundles
        """
        let iron = input("iron-plate", 0);
        let copper = input("copper-plate", 1);
        let resources = bundle(iron, copper);
        let output = resources | "steel-plate";
        """,
        
        # Functions
        """
        func double(x) {
            return x * 2;
        }
        let result = double(input(0));
        """,
        
        # Entity placement
        """
        let lamp = Place("small-lamp", 0, 0);
        lamp.enable = input(0) > 0;
        """
    ]
    
    for i, test_code in enumerate(test_cases):
        print(f"\n=== Test Case {i+1} ===")
        print(test_code.strip())
        
        try:
            ast = parser.parse(test_code)
            print("✓ Parse successful!")
        except Exception as e:
            print(f"✗ Parse failed: {e}")


def test_file_parsing():
    """Legacy test function for backward compatibility."""
    parser = DSLParser()
    test_dir = Path(__file__).parent.parent / "dsl_compiler" / "tests"
    
    if not test_dir.exists():
        print("Test directory not found")
        return
    
    test_files = list(test_dir.glob("*.fcdsl"))
    print(f"\nFound {len(test_files)} test files")
    
    for test_file in test_files[:3]:  # Test first 3 files
        print(f"\n=== Testing {test_file.name} ===")
        try:
            ast = parser.parse_file(test_file)
            print(f"✓ {test_file.name} parsed successfully!")
            print(f"  Contains {len(ast.statements)} statements")
        except Exception as e:
            print(f"✗ {test_file.name} failed: {e}")


if __name__ == "__main__":
    print("Testing DSL Parser")
    print("=" * 50)
    
    test_basic_parsing()
    test_file_parsing()
    
    print("\n" + "=" * 50)
    print("Parser testing complete")
