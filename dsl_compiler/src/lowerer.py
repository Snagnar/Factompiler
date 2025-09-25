# lowerer.py
"""
AST to IR lowering pass for the Factorio Circuit DSL.

This module converts semantic-analyzed AST nodes into IR operations
following the lowering rules specified in the compiler specification.
"""

from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

from dsl_compiler.src.dsl_ast import *
from dsl_compiler.src.ir import *
from dsl_compiler.src.semantic import (
    SemanticAnalyzer, SignalValue, BundleValue, IntValue, 
    ValueInfo, DiagnosticCollector
)


class ASTLowerer:
    """Converts semantic-analyzed AST to IR."""
    
    def __init__(self, semantic_analyzer: SemanticAnalyzer):
        self.semantic = semantic_analyzer
        self.ir_builder = IRBuilder()
        self.diagnostics = DiagnosticCollector()
        
        # Symbol tables for IR references
        self.signal_refs: Dict[str, SignalRef] = {}  # variable_name -> SignalRef
        self.bundle_refs: Dict[str, BundleRef] = {}  # variable_name -> BundleRef
        self.memory_refs: Dict[str, str] = {}        # memory_name -> memory_id
        self.entity_refs: Dict[str, str] = {}        # entity_name -> entity_id
        
        # Copy signal type mapping from semantic analyzer
        self.ir_builder.signal_type_map = self.semantic.signal_type_map.copy()
        
    def lower_program(self, program: Program) -> List[IRNode]:
        """Lower an entire program to IR."""
        for stmt in program.statements:
            self.lower_statement(stmt)
        
        return self.ir_builder.get_ir()
    
    def lower_statement(self, stmt: Statement):
        """Lower a statement to IR."""
        if isinstance(stmt, LetStmt):
            self.lower_let_stmt(stmt)
        elif isinstance(stmt, AssignStmt):
            self.lower_assign_stmt(stmt)
        elif isinstance(stmt, MemDecl):
            self.lower_mem_decl(stmt)
        elif isinstance(stmt, ExprStmt):
            self.lower_expr_stmt(stmt)
        elif isinstance(stmt, ReturnStmt):
            self.lower_return_stmt(stmt)
        elif isinstance(stmt, FuncDecl):
            self.lower_func_decl(stmt)
        elif isinstance(stmt, ImportStmt):
            self.lower_import_stmt(stmt)
        else:
            self.diagnostics.warning(f"Unknown statement type: {type(stmt)}", stmt)
    
    def lower_let_stmt(self, stmt: LetStmt):
        """Lower let statement: let name = expr;"""
        value_ref = self.lower_expr(stmt.value)
        
        # Store the reference for later use
        if isinstance(value_ref, SignalRef):
            self.signal_refs[stmt.name] = value_ref
        elif isinstance(value_ref, BundleRef):
            self.bundle_refs[stmt.name] = value_ref
        elif isinstance(value_ref, int):
            # Convert integer to constant signal
            # Get the expected type from semantic analysis
            symbol = self.semantic.symbol_table.lookup(stmt.name)
            if symbol and isinstance(symbol.value_type, SignalValue):
                signal_type = symbol.value_type.signal_type.name
            else:
                signal_type = self.ir_builder.allocate_implicit_type()
            
            const_ref = self.ir_builder.const(signal_type, value_ref, stmt)
            self.signal_refs[stmt.name] = const_ref
    
    def lower_assign_stmt(self, stmt: AssignStmt):
        """Lower assignment statement: target = expr;"""
        value_ref = self.lower_expr(stmt.value)
        
        if isinstance(stmt.target, Identifier):
            # Simple variable assignment
            if isinstance(value_ref, SignalRef):
                self.signal_refs[stmt.target.name] = value_ref
            elif isinstance(value_ref, BundleRef):
                self.bundle_refs[stmt.target.name] = value_ref
            
        elif isinstance(stmt.target, PropertyAccess):
            # Entity property assignment
            entity_name = stmt.target.object_name
            prop_name = stmt.target.property_name
            
            if entity_name in self.entity_refs:
                entity_id = self.entity_refs[entity_name]
                self.ir_builder.add_operation(
                    IR_EntityPropWrite(f"prop_write_{entity_id}_{prop_name}", stmt)
                )
                self.ir_builder.operations[-1].entity_id = entity_id
                self.ir_builder.operations[-1].property_name = prop_name
                self.ir_builder.operations[-1].value = value_ref
    
    def lower_mem_decl(self, stmt: MemDecl):
        """Lower memory declaration: mem name = memory(init);"""
        memory_id = f"mem_{stmt.name}"
        self.memory_refs[stmt.name] = memory_id
        
        # Get memory type from semantic analysis
        symbol = self.semantic.symbol_table.lookup(stmt.name)
        if symbol and isinstance(symbol.value_type, SignalValue):
            signal_type = symbol.value_type.signal_type.name
        else:
            signal_type = self.ir_builder.allocate_implicit_type()
        
        # Lower initialization expression if present
        if stmt.init_expr:
            init_ref = self.lower_expr(stmt.init_expr)
            if isinstance(init_ref, int):
                init_ref = self.ir_builder.const(signal_type, init_ref, stmt)
        else:
            init_ref = self.ir_builder.const(signal_type, 0, stmt)
        
        self.ir_builder.memory_create(memory_id, signal_type, init_ref, stmt)
    
    def lower_expr_stmt(self, stmt: ExprStmt):
        """Lower expression statement: expr;"""
        self.lower_expr(stmt.expr)
    
    def lower_return_stmt(self, stmt: ReturnStmt):
        """Lower return statement: return expr;"""
        if stmt.expr:
            return_ref = self.lower_expr(stmt.expr)
            # For now, just lower the expression
            # TODO: Handle function return values properly
    
    def lower_func_decl(self, stmt: FuncDecl):
        """Lower function declaration."""
        # TODO: Implement function lowering with IR_Group
        self.diagnostics.warning(f"Function lowering not yet implemented: {stmt.name}", stmt)
    
    def lower_import_stmt(self, stmt: ImportStmt):
        """Lower import statement."""
        # TODO: Implement module imports
        self.diagnostics.warning(f"Import lowering not yet implemented: {stmt.path}", stmt)
    
    def lower_expr(self, expr: Expr) -> ValueRef:
        """Lower an expression to IR, returning a ValueRef."""
        if isinstance(expr, NumberLiteral):
            return expr.value
            
        elif isinstance(expr, StringLiteral):
            # String literals are only used as type specifiers, not values
            self.diagnostics.error("String literals cannot be used as values", expr)
            return 0
            
        elif isinstance(expr, IdentifierExpr):
            return self.lower_identifier(expr)
            
        elif isinstance(expr, BinaryOp):
            return self.lower_binary_op(expr)
            
        elif isinstance(expr, UnaryOp):
            return self.lower_unary_op(expr)
            
        elif isinstance(expr, InputExpr):
            return self.lower_input_expr(expr)
            
        elif isinstance(expr, ReadExpr):
            return self.lower_read_expr(expr)
            
        elif isinstance(expr, WriteExpr):
            return self.lower_write_expr(expr)
            
        elif isinstance(expr, BundleExpr):
            return self.lower_bundle_expr(expr)
            
        elif isinstance(expr, ProjectionExpr):
            return self.lower_projection_expr(expr)
            
        elif isinstance(expr, CallExpr):
            return self.lower_call_expr(expr)
            
        else:
            self.diagnostics.error(f"Unknown expression type: {type(expr)}", expr)
            return 0
    
    def lower_identifier(self, expr: IdentifierExpr) -> ValueRef:
        """Lower identifier reference."""
        name = expr.name
        
        if name in self.signal_refs:
            return self.signal_refs[name]
        elif name in self.bundle_refs:
            return self.bundle_refs[name]
        else:
            self.diagnostics.error(f"Undefined identifier: {name}", expr)
            # Return a dummy constant
            return self.ir_builder.const(self.ir_builder.allocate_implicit_type(), 0, expr)
    
    def lower_binary_op(self, expr: BinaryOp) -> ValueRef:
        """Lower binary operation following mixed-type rules."""
        left_ref = self.lower_expr(expr.left)
        right_ref = self.lower_expr(expr.right)
        
        # Get type information from semantic analysis
        left_type = self.semantic.get_expr_type(expr.left)
        right_type = self.semantic.get_expr_type(expr.right)
        result_type = self.semantic.get_expr_type(expr)
        
        # Determine output signal type
        if isinstance(result_type, SignalValue):
            output_type = result_type.signal_type.name
        elif isinstance(result_type, IntValue):
            # This is a comparison operation, return integer
            return self.lower_comparison_op(expr, left_ref, right_ref)
        else:
            output_type = self.ir_builder.allocate_implicit_type()
        
        # Handle different operand combinations
        if expr.op in ["+", "-", "*", "/", "%"]:
            return self.ir_builder.arithmetic(expr.op, left_ref, right_ref, output_type, expr)
        else:
            # This should be handled by comparison lowering
            return self.lower_comparison_op(expr, left_ref, right_ref)
    
    def lower_comparison_op(self, expr: BinaryOp, left_ref: ValueRef, right_ref: ValueRef) -> ValueRef:
        """Lower comparison operations to decider combinators."""
        # Get output type for the boolean result
        output_type = self.ir_builder.allocate_implicit_type()
        
        if expr.op in ["==", "!=", "<", "<=", ">", ">="]:
            # Use decider combinator
            return self.ir_builder.decider(expr.op, left_ref, right_ref, 1, output_type, expr)
        elif expr.op == "&&":
            # Logical AND: both operands must be non-zero
            # This is more complex, but for now use arithmetic multiplication
            return self.ir_builder.arithmetic("*", left_ref, right_ref, output_type, expr)
        elif expr.op == "||":
            # Logical OR: either operand must be non-zero
            # Use addition with a decider to clamp to 0/1
            sum_ref = self.ir_builder.arithmetic("+", left_ref, right_ref, output_type, expr)
            # If sum > 0, output 1, else 0
            return self.ir_builder.decider(">", sum_ref, 0, 1, output_type, expr)
        else:
            self.diagnostics.error(f"Unknown binary operator: {expr.op}", expr)
            return self.ir_builder.const(output_type, 0, expr)
    
    def lower_unary_op(self, expr: UnaryOp) -> ValueRef:
        """Lower unary operation."""
        operand_ref = self.lower_expr(expr.expr)
        
        # Get output type from semantic analysis
        result_type = self.semantic.get_expr_type(expr)
        if isinstance(result_type, SignalValue):
            output_type = result_type.signal_type.name
        else:
            output_type = self.ir_builder.allocate_implicit_type()
        
        if expr.op == "+":
            # Unary plus is a no-op
            return operand_ref
        elif expr.op == "-":
            # Unary minus: multiply by -1
            neg_one = self.ir_builder.const(output_type, -1, expr)
            return self.ir_builder.arithmetic("*", operand_ref, neg_one, output_type, expr)
        elif expr.op == "!":
            # Logical not: if operand == 0 then 1 else 0
            return self.ir_builder.decider("==", operand_ref, 0, 1, output_type, expr)
        else:
            self.diagnostics.error(f"Unknown unary operator: {expr.op}", expr)
            return operand_ref
    
    def lower_input_expr(self, expr: InputExpr) -> SignalRef:
        """Lower input expression: input(index) or input(type, index)."""
        if expr.signal_type:
            # Explicit type specified
            if isinstance(expr.signal_type, StringLiteral):
                signal_type = expr.signal_type.value
            else:
                signal_type = str(expr.signal_type)
        else:
            # Allocate implicit type
            signal_type = self.ir_builder.allocate_implicit_type()
        
        return self.ir_builder.input_signal(expr.index, signal_type, expr)
    
    def lower_read_expr(self, expr: ReadExpr) -> SignalRef:
        """Lower memory read: read(memory)."""
        memory_name = expr.memory_name
        
        if memory_name not in self.memory_refs:
            self.diagnostics.error(f"Undefined memory: {memory_name}", expr)
            signal_type = self.ir_builder.allocate_implicit_type()
            return self.ir_builder.const(signal_type, 0, expr)
        
        memory_id = self.memory_refs[memory_name]
        
        # Get memory type from semantic analysis
        symbol = self.semantic.symbol_table.lookup(memory_name)
        if symbol and isinstance(symbol.value_type, SignalValue):
            signal_type = symbol.value_type.signal_type.name
        else:
            signal_type = self.ir_builder.allocate_implicit_type()
        
        return self.ir_builder.memory_read(memory_id, signal_type, expr)
    
    def lower_write_expr(self, expr: WriteExpr) -> SignalRef:
        """Lower memory write: write(memory, value)."""
        memory_name = expr.memory_name
        
        if memory_name not in self.memory_refs:
            self.diagnostics.error(f"Undefined memory: {memory_name}", expr)
            return self.ir_builder.const(self.ir_builder.allocate_implicit_type(), 0, expr)
        
        memory_id = self.memory_refs[memory_name]
        data_ref = self.lower_expr(expr.value)
        
        # Write enable is always 1 (unconditional write)
        write_enable = self.ir_builder.const(self.ir_builder.allocate_implicit_type(), 1, expr)
        
        self.ir_builder.memory_write(memory_id, data_ref, write_enable, expr)
        
        # Write expressions return the written value
        return data_ref if isinstance(data_ref, SignalRef) else self.ir_builder.const(self.ir_builder.allocate_implicit_type(), 0, expr)
    
    def lower_bundle_expr(self, expr: BundleExpr) -> BundleRef:
        """Lower bundle expression: bundle(expr1, expr2, ...)."""
        inputs = {}
        
        for sub_expr in expr.exprs:
            sub_ref = self.lower_expr(sub_expr)
            
            if isinstance(sub_ref, SignalRef):
                inputs[sub_ref.signal_type] = sub_ref
            elif isinstance(sub_ref, BundleRef):
                # Flatten bundle into inputs
                for sig_type, sig_ref in sub_ref.channels.items():
                    inputs[sig_type] = sig_ref
            # Ignore integer values in bundles
        
        return self.ir_builder.bundle(inputs, expr)
    
    def lower_projection_expr(self, expr: ProjectionExpr) -> SignalRef:
        """Lower projection: expr | "type"."""
        source_ref = self.lower_expr(expr.expr)
        target_type = expr.target_type
        
        if isinstance(source_ref, SignalRef):
            if source_ref.signal_type == target_type:
                # No-op projection to same type
                return source_ref
            else:
                # Convert signal to target type using arithmetic passthrough
                return self.ir_builder.arithmetic("+", source_ref, 0, target_type, expr)
        
        elif isinstance(source_ref, BundleRef):
            if target_type in source_ref.channels:
                # Extract specific channel
                return source_ref.channels[target_type]
            else:
                # Sum all channels into target type
                self.diagnostics.warning(f"Projection collapsing bundle to {target_type}", expr)
                # For now, just create a constant - proper summing would need more IR nodes
                return self.ir_builder.const(target_type, 0, expr)
        
        elif isinstance(source_ref, int):
            # Convert integer to signal of target type
            return self.ir_builder.const(target_type, source_ref, expr)
        
        else:
            self.diagnostics.error(f"Cannot project {type(source_ref)} to {target_type}", expr)
            return self.ir_builder.const(target_type, 0, expr)
    
    def lower_call_expr(self, expr: CallExpr) -> ValueRef:
        """Lower function call or special calls like Place()."""
        if expr.name == "Place":
            return self.lower_place_call(expr)
        else:
            # Regular function call
            self.diagnostics.warning(f"Function calls not yet implemented: {expr.name}", expr)
            return self.ir_builder.const(self.ir_builder.allocate_implicit_type(), 0, expr)
    
    def lower_place_call(self, expr: CallExpr) -> SignalRef:
        """Lower Place() call for entity placement."""
        if len(expr.args) < 3:
            self.diagnostics.error("Place() requires at least 3 arguments: (prototype, x, y)", expr)
            return self.ir_builder.const(self.ir_builder.allocate_implicit_type(), 0, expr)
        
        # Extract arguments
        prototype_expr = expr.args[0]
        x_expr = expr.args[1]
        y_expr = expr.args[2]
        
        if not isinstance(prototype_expr, StringLiteral):
            self.diagnostics.error("Place() prototype must be a string literal", prototype_expr)
            return self.ir_builder.const(self.ir_builder.allocate_implicit_type(), 0, expr)
        
        prototype = prototype_expr.value
        
        # Evaluate position arguments
        x_ref = self.lower_expr(x_expr)
        y_ref = self.lower_expr(y_expr)
        
        if not isinstance(x_ref, int) or not isinstance(y_ref, int):
            self.diagnostics.error("Place() coordinates must be integer constants", expr)
            return self.ir_builder.const(self.ir_builder.allocate_implicit_type(), 0, expr)
        
        # Generate unique entity ID
        entity_id = f"entity_{self.ir_builder.next_id()}"
        
        # Place the entity
        self.ir_builder.place_entity(entity_id, prototype, x_ref, y_ref, source_ast=expr)
        
        # Return a dummy signal (entities don't produce signals directly)
        return self.ir_builder.const(self.ir_builder.allocate_implicit_type(), 0, expr)


# =============================================================================
# Public API
# =============================================================================

def lower_program(program: Program, semantic_analyzer: SemanticAnalyzer) -> tuple[List[IRNode], DiagnosticCollector]:
    """Lower a semantic-analyzed program to IR."""
    lowerer = ASTLowerer(semantic_analyzer)
    ir_operations = lowerer.lower_program(program)
    return ir_operations, lowerer.diagnostics


if __name__ == "__main__":
    # Test IR lowering
    from dsl_compiler.src.parser import DSLParser
    from dsl_compiler.src.semantic import analyze_program
    
    test_code = """
    let a = input(0);
    let b = input("iron-plate", 1);
    let sum = a + b;
    let result = sum | "signal-output";
    
    mem counter = memory(0);
    let count = read(counter);
    write(counter, count + 1);
    """
    
    parser = DSLParser()
    try:
        program = parser.parse(test_code)
        
        # Run semantic analysis
        from dsl_compiler.src.semantic import SemanticAnalyzer
        analyzer = SemanticAnalyzer()
        semantic_diagnostics = analyze_program(program, strict_types=False, analyzer=analyzer)
        if semantic_diagnostics.has_errors():
            print("Semantic errors:")
            for msg in semantic_diagnostics.get_messages():
                print(f"  {msg}")
        else:
            # Lower to IR using the analyzer that already processed the program
            ir_operations, lowering_diagnostics = lower_program(program, analyzer)
            
            print("IR Lowering Results:")
            print("=" * 50)
            
            for i, op in enumerate(ir_operations):
                print(f"{i:2d}: {op}")
            
            print("\nDiagnostics:")
            for msg in lowering_diagnostics.get_messages():
                print(f"  {msg}")
            
            print(f"\nGenerated {len(ir_operations)} IR operations")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
