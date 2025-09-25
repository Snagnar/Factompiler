# test_parser_suite.py
"""
Comprehensive pytest test suite for the DSL parser.
Tests parsing of all example programs in the tests/ directory.
"""

import pytest
from pathlib import Path

from dsl_compiler.src.parser import DSLParser
from dsl_compiler.src.dsl_ast import Program, print_ast


class TestParserSuite:
    """Test suite for DSL parser functionality."""
    
    @classmethod
    def setup_class(cls):
        """Set up test class."""
        cls.parser = DSLParser()
        cls.test_dir = Path(__file__).parent / "sample_programs"
        
    def test_parser_initialization(self):
        """Test that parser initializes correctly."""
        assert self.parser is not None
        assert self.parser.parser is not None
        
    @pytest.mark.parametrize("test_file", [
        "01_basic_arithmetic.fcdsl",
        "02_mixed_types.fcdsl", 
        "03_bundles.fcdsl",
        "04_memory.fcdsl",
        "05_entities.fcdsl",
        "06_functions.fcdsl",
        "07_control_flow.fcdsl",
        "08_sampler.fcdsl",
        "09_advanced_patterns.fcdsl",
        "10_imports_modules.fcdsl",
        "11_edge_cases.fcdsl"
    ])
    def test_parse_test_files(self, test_file):
        """Test parsing of individual test files."""
        file_path = self.test_dir / test_file
        
        if not file_path.exists():
            pytest.skip(f"Test file {test_file} not found")
            
        try:
            program = self.parser.parse_file(file_path)
            assert isinstance(program, Program)
            assert len(program.statements) > 0
            print(f"✓ {test_file}: {len(program.statements)} statements parsed")
            
        except Exception as e:
            pytest.fail(f"Failed to parse {test_file}: {e}")
            
    def test_basic_expressions(self):
        """Test basic expression parsing."""
        test_cases = [
            ("let a = 42;", "number literal"),
            ("let b = input(0);", "input without type"),
            ("let c = input(\"iron-plate\", 1);", "input with type"),
            ("let d = a + b;", "binary addition"),
            ("let e = -a;", "unary minus"),
            ("let f = a * 2 + b;", "precedence"),
        ]
        
        for code, description in test_cases:
            try:
                program = self.parser.parse(code)
                assert isinstance(program, Program)
                assert len(program.statements) == 1
                print(f"✓ {description}")
            except Exception as e:
                pytest.fail(f"Failed to parse {description}: {code}\nError: {e}")
                
    def test_memory_operations(self):
        """Test memory-related parsing."""
        test_cases = [
            ("mem counter = memory();", "memory without init"),
            ("mem total = memory(0);", "memory with init"),
            ("let val = read(counter);", "memory read"),
            ("write(counter, val + 1);", "memory write"),
        ]
        
        for code, description in test_cases:
            try:
                program = self.parser.parse(code)
                assert isinstance(program, Program)
                print(f"✓ {description}")
            except Exception as e:
                pytest.fail(f"Failed to parse {description}: {code}\nError: {e}")
                
    def test_projection_and_bundles(self):
        """Test projection and bundle parsing."""
        test_cases = [
            ("let a = input(0) | \"iron-plate\";", "simple projection"),
            ("let b = bundle(a, b);", "simple bundle"),
            ("let c = bundle(a, b) | \"steel\";", "bundle projection"),
        ]
        
        for code, description in test_cases:
            try:
                program = self.parser.parse(code)
                assert isinstance(program, Program)
                print(f"✓ {description}")
            except Exception as e:
                pytest.fail(f"Failed to parse {description}: {code}\nError: {e}")
                
    def test_functions(self):
        """Test function parsing."""
        code = """
        func double(x) {
            return x * 2;
        }
        let result = double(5);
        """
        
        try:
            program = self.parser.parse(code)
            assert isinstance(program, Program)
            assert len(program.statements) == 2  # function + let
            print("✓ Function definition and call")
        except Exception as e:
            pytest.fail(f"Failed to parse function: {e}")
            
    def test_entity_placement(self):
        """Test entity placement parsing."""
        code = """
        let lamp = Place("small-lamp", 0, 0);
        lamp.enable = input(0) > 0;
        """
        
        try:
            program = self.parser.parse(code)
            assert isinstance(program, Program)
            print("✓ Entity placement and property assignment")
        except Exception as e:
            pytest.fail(f"Failed to parse entity placement: {e}")


def run_parser_diagnostics():
    """Run diagnostic parsing on all test files and report results."""
    parser = DSLParser()
    test_dir = Path(__file__).parent / "dsl_compiler" / "tests"
    
    if not test_dir.exists():
        print(f"Test directory not found: {test_dir}")
        return
        
    test_files = sorted(test_dir.glob("*.fcdsl"))
    print(f"Found {len(test_files)} test files")
    print("=" * 60)
    
    passed = 0
    total = len(test_files)
    
    for test_file in test_files:
        print(f"\nTesting {test_file.name}...")
        try:
            program = parser.parse_file(test_file)
            print(f"✓ SUCCESS: {len(program.statements)} statements parsed")
            passed += 1
            
            # Show first few statements for verification
            for i, stmt in enumerate(program.statements[:3]):
                print(f"  {i+1}: {type(stmt).__name__}")
            if len(program.statements) > 3:
                print(f"  ... and {len(program.statements) - 3} more")
                
        except Exception as e:
            print(f"✗ FAILED: {e}")
            
            # Try to get more details about the error
            try:
                with open(test_file, 'r') as f:
                    content = f.read()
                lines = content.strip().split('\n')
                print(f"  File has {len(lines)} lines")
                print(f"  First line: {lines[0] if lines else '(empty)'}")
            except:
                pass
                
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} files parsed successfully")
    
    if passed < total:
        print(f"\nNeed to fix {total - passed} parser issues")
        return False
    else:
        print("\n🎉 All test files parsed successfully!")
        return True


if __name__ == "__main__":
    print("DSL Parser Test Suite")
    print("=" * 60)
    
    # Run diagnostics first
    success = run_parser_diagnostics()
    
    if not success:
        print("\nRun with pytest for detailed test results:")
        print("  pytest test_parser_suite.py -v")
    else:
        print("\nReady to run full test suite:")
        print("  pytest test_parser_suite.py -v")
