# test_parser.py
"""
Test script for the DSL parser.
"""

import sys
from pathlib import Path

from dsl_compiler.src.parser import DSLParser
from dsl_compiler.src.dsl_ast import print_ast


def test_basic_parsing():
    """Test basic parsing functionality."""
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
            # Uncomment to see AST structure:
            # print_ast(ast)
        except Exception as e:
            print(f"✗ Parse failed: {e}")


def test_file_parsing():
    """Test parsing actual test files."""
    parser = DSLParser()
    test_dir = Path(__file__).parent / "sample_programs"
    
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
