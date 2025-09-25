#!/usr/bin/env python3
"""
Simple test of AST classes without requiring lark.
"""

import sys
from pathlib import Path

from dsl_compiler.src.dsl_ast import *

def test_ast_classes():
    """Test AST class construction."""
    print("Testing AST class construction...")
    
    # Test basic expressions
    num_lit = NumberLiteral(value=42)
    str_lit = StringLiteral(value="iron-plate")
    ident = IdentifierExpr(name="my_var")
    
    print(f"✓ Number literal: {num_lit}")
    print(f"✓ String literal: {str_lit}")
    print(f"✓ Identifier: {ident}")
    
    # Test binary operation
    binop = BinaryOp(op="+", left=num_lit, right=ident)
    print(f"✓ Binary op: {binop}")
    
    # Test statements
    let_stmt = LetStmt(name="result", value=binop)
    input_expr = InputExpr(index=NumberLiteral(value=0))
    mem_decl = MemDecl(name="counter", init_expr=NumberLiteral(value=0))
    
    print(f"✓ Let statement: {let_stmt}")
    print(f"✓ Input expression: {input_expr}")
    print(f"✓ Memory declaration: {mem_decl}")
    
    # Test projection
    proj = ProjectionExpr(expr=ident, target_type="signal-output")
    print(f"✓ Projection: {proj}")
    
    # Test bundle
    bundle = BundleExpr(exprs=[num_lit, str_lit])
    print(f"✓ Bundle: {bundle}")
    
    # Test program
    program = Program(statements=[let_stmt, mem_decl])
    print(f"✓ Program with {len(program.statements)} statements")
    
    print("\n=== AST Tree Structure ===")
    print_ast(program)
    
    print("\n=== AST Dictionary Representation ===")
    ast_dict = ast_to_dict(program)
    import json
    print(json.dumps(ast_dict, indent=2))


def test_draftsman_availability():
    """Test if factorio-draftsman is available."""
    print("\nTesting factorio-draftsman availability...")
    
    try:
        sys.path.insert(0, "/home/paul/projects/factorio-draftsman")
        from draftsman.blueprintable import Blueprint
        from draftsman.entity import ConstantCombinator
        
        print("✓ factorio-draftsman imported successfully!")
        
        # Test basic blueprint creation
        bp = Blueprint()
        bp.label = "Test Blueprint"
        bp.version = (2, 0)
        
        # Test adding a combinator
        combinator = ConstantCombinator(tile_position=(0, 0))
        bp.entities.append(combinator)
        
        print(f"✓ Created blueprint with {len(bp.entities)} entities")
        
        return True
        
    except ImportError as e:
        print(f"✗ factorio-draftsman not available: {e}")
        return False


if __name__ == "__main__":
    print("DSL Compiler Component Tests")
    print("=" * 50)
    
    test_ast_classes()
    
    draftsman_ok = test_draftsman_availability()
    
    print("\n" + "=" * 50)
    print("Component testing complete")
    
    if not draftsman_ok:
        print("\nNote: Parser requires lark installation")
        print("      Blueprint generation requires factorio-draftsman")
        print("      Install with: pip install lark")
