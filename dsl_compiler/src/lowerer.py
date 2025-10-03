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
    SemanticAnalyzer,
    SignalValue,
    BundleValue,
    IntValue,
    ValueInfo,
    DiagnosticCollector,
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
        self.memory_refs: Dict[str, str] = {}  # memory_name -> memory_id
        self.entity_refs: Dict[str, str] = {}  # entity_name -> entity_id
        self.param_values: Dict[str, ValueRef] = {}  # parameter_name -> value during function inlining

        # Copy signal type mapping from semantic analyzer
        self.ir_builder.signal_type_map = self.semantic.signal_type_map.copy()

    def lower_program(self, program: Program) -> List[IRNode]:
        """Lower an entire program to IR."""
        for stmt in program.statements:
            self.lower_statement(stmt)

        return self.ir_builder.get_ir()

    def lower_statement(self, stmt: Statement):
        """Lower a statement to IR."""
        # Dispatch table for statement lowering
        statement_handlers = {
            DeclStmt: self.lower_decl_stmt,
            AssignStmt: self.lower_assign_stmt,
            MemDecl: self.lower_mem_decl,
            ExprStmt: self.lower_expr_stmt,
            ReturnStmt: self.lower_return_stmt,
            FuncDecl: self.lower_func_decl,
            ImportStmt: self.lower_import_stmt,
        }

        handler = statement_handlers.get(type(stmt))
        if handler:
            handler(stmt)
        else:
            self.diagnostics.error(f"Unknown statement type: {type(stmt)}", stmt)

    def lower_decl_stmt(self, stmt: DeclStmt):
        """Lower typed declaration statement: type name = expr;"""
        # Handle Memory type declarations
        if stmt.type_name == "Memory":
            return self.lower_memory_decl(stmt)
        
        # Special handling for place() calls to track entities
        if isinstance(stmt.value, CallExpr) and stmt.value.name == "place":
            entity_id, value_ref = self.lower_place_call_with_tracking(stmt.value)
            self.entity_refs[stmt.name] = entity_id
            self.signal_refs[stmt.name] = value_ref
            return
        
        # Special handling for function calls that return entities
        if isinstance(stmt.value, CallExpr):
            # Clear any returned entity from previous call
            self.returned_entity_id = None
            value_ref = self.lower_expr(stmt.value)
            
            # If the function call returned an entity, track it
            if hasattr(self, 'returned_entity_id') and self.returned_entity_id is not None:
                self.entity_refs[stmt.name] = self.returned_entity_id
                self.returned_entity_id = None
            
            self.signal_refs[stmt.name] = value_ref
            return

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
        if isinstance(stmt.target, Identifier):
            # Special handling for place() calls to track entities
            if isinstance(stmt.value, CallExpr) and stmt.value.name == "place":
                entity_id, value_ref = self.lower_place_call_with_tracking(stmt.value)
                self.entity_refs[stmt.target.name] = entity_id
                self.signal_refs[stmt.target.name] = value_ref
                return
            
            # Special handling for function calls that return entities
            if isinstance(stmt.value, CallExpr):
                # Clear any returned entity from previous call
                self.returned_entity_id = None
                value_ref = self.lower_expr(stmt.value)
                
                # If the function call returned an entity, track it
                if hasattr(self, 'returned_entity_id') and self.returned_entity_id is not None:
                    self.entity_refs[stmt.target.name] = self.returned_entity_id
                    self.returned_entity_id = None
                
                self.signal_refs[stmt.target.name] = value_ref
                return

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
                prop_write_op = IR_EntityPropWrite(entity_id, prop_name, value_ref)
                self.ir_builder.add_operation(prop_write_op)

    def lower_mem_decl(self, stmt: MemDecl):
        """Lower memory declaration: Memory name = init;"""
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

    def lower_memory_decl(self, stmt: DeclStmt):
        """Lower Memory type declaration: Memory name = init_expr;"""
        memory_id = f"mem_{stmt.name}"
        self.memory_refs[stmt.name] = memory_id

        # Get memory type from semantic analysis
        symbol = self.semantic.symbol_table.lookup(stmt.name)
        if symbol and isinstance(symbol.value_type, SignalValue):
            signal_type = symbol.value_type.signal_type.name
        else:
            signal_type = self.ir_builder.allocate_implicit_type()

        # Lower initialization expression
        init_ref = self.lower_expr(stmt.value)
        if isinstance(init_ref, int):
            init_ref = self.ir_builder.const(signal_type, init_ref, stmt)

        self.ir_builder.memory_create(memory_id, signal_type, init_ref, stmt)

    def lower_expr_stmt(self, stmt: ExprStmt):
        """Lower expression statement: expr;"""
        self.lower_expr(stmt.expr)

    def lower_return_stmt(self, stmt: ReturnStmt):
        """Lower return statement: return expr;"""
        if stmt.expr:
            return_ref = self.lower_expr(stmt.expr)
            # Store the return value for function inlining
            # The function inlining mechanism will pick this up
            return return_ref
        return None

    def lower_func_decl(self, stmt: FuncDecl):
        """Lower function declaration."""
        # Store function definition for later inlining - don't generate IR yet
        # Functions will be inlined when called
        pass

    def lower_import_stmt(self, stmt: ImportStmt):
        """Lower import statement."""
        # Import statements should have been preprocessed and inlined by the parser.
        # If we encounter one here, it means the file wasn't found during preprocessing.
        self.diagnostics.error(
            f"Import statement found in AST - file not found during preprocessing: {stmt.path}", stmt
        )

    def lower_expr(self, expr: Expr) -> ValueRef:
        """Lower an expression to IR, returning a ValueRef."""
        # Special cases that return immediate values
        if isinstance(expr, NumberLiteral):
            return expr.value
        elif isinstance(expr, StringLiteral):
            # String literals are valid values - they represent strings
            return expr.value

        # Dispatch table for expression lowering
        expression_handlers = {
            IdentifierExpr: self.lower_identifier,
            BinaryOp: self.lower_binary_op,
            UnaryOp: self.lower_unary_op,
            ReadExpr: self.lower_read_expr,
            WriteExpr: self.lower_write_expr,
            BundleExpr: self.lower_bundle_expr,
            ProjectionExpr: self.lower_projection_expr,
            CallExpr: self.lower_call_expr,
            PropertyAccess: self.lower_property_access,
            PropertyAccessExpr: self.lower_property_access_expr,
            SignalLiteral: self.lower_signal_literal,
        }

        handler = expression_handlers.get(type(expr))
        if handler:
            return handler(expr)
        else:
            self.diagnostics.error(f"Unknown expression type: {type(expr)}", expr)
            return 0

    def lower_identifier(self, expr: IdentifierExpr) -> ValueRef:
        """Lower identifier reference."""
        name = expr.name

        # Check parameter values first (for function inlining)
        if name in self.param_values:
            return self.param_values[name]
        elif name in self.signal_refs:
            return self.signal_refs[name]
        elif name in self.bundle_refs:
            return self.bundle_refs[name]
        else:
            self.diagnostics.error(f"Undefined identifier: {name}", expr)
            return self.ir_builder.const(self.ir_builder.allocate_implicit_type(), 0, expr)

    def lower_binary_op(self, expr: BinaryOp) -> ValueRef:
        """Lower binary operation following mixed-type rules."""
        left_ref = self.lower_expr(expr.left)
        right_ref = self.lower_expr(expr.right)

        # Get type information from semantic analysis
        left_type = self.semantic.get_expr_type(expr.left)
        right_type = self.semantic.get_expr_type(expr.right)
        result_type = self.semantic.get_expr_type(expr)

        # Determine output signal type based on result type
        if isinstance(result_type, SignalValue):
            output_type = result_type.signal_type.name
        elif isinstance(result_type, IntValue):
            # Integer result - use an implicit type for the signal
            output_type = self.ir_builder.allocate_implicit_type()
        else:
            output_type = self.ir_builder.allocate_implicit_type()

        # Handle different operand combinations
        if expr.op in ["+", "-", "*", "/", "%"]:
            return self.ir_builder.arithmetic(
                expr.op, left_ref, right_ref, output_type, expr
            )
        elif expr.op in ["==", "!=", "<", "<=", ">", ">=", "&&", "||"]:
            # This should be handled by comparison lowering
            return self.lower_comparison_op(expr, left_ref, right_ref)
        else:
            self.diagnostics.error(f"Unknown binary operator: {expr.op}", expr)
            return self.ir_builder.const(output_type, 0, expr)

    def lower_comparison_op(
        self, expr: BinaryOp, left_ref: ValueRef, right_ref: ValueRef
    ) -> ValueRef:
        """Lower comparison operations to decider combinators."""
        # Get output type for the boolean result
        output_type = self.ir_builder.allocate_implicit_type()

        if expr.op in ["==", "!=", "<", "<=", ">", ">="]:
            # Use decider combinator
            return self.ir_builder.decider(
                expr.op, left_ref, right_ref, 1, output_type, expr
            )
        elif expr.op == "&&":
            # Logical AND: both operands must be non-zero
            # This is more complex, but for now use arithmetic multiplication
            return self.ir_builder.arithmetic(
                "*", left_ref, right_ref, output_type, expr
            )
        elif expr.op == "||":
            # Logical OR: either operand must be non-zero
            # Use addition with a decider to clamp to 0/1
            sum_ref = self.ir_builder.arithmetic(
                "+", left_ref, right_ref, output_type, expr
            )
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
            return self.ir_builder.arithmetic(
                "*", operand_ref, neg_one, output_type, expr
            )
        elif expr.op == "!":
            # Logical not: if operand == 0 then 1 else 0
            return self.ir_builder.decider("==", operand_ref, 0, 1, output_type, expr)
        else:
            self.diagnostics.error(f"Unknown unary operator: {expr.op}", expr)
            return operand_ref

    def lower_signal_literal(self, expr: SignalLiteral) -> SignalRef:
        """Lower signal literal: ("type", value) or just value."""
        if expr.signal_type:
            # Explicit type specified
            signal_type = expr.signal_type
        else:
            # Allocate implicit type
            signal_type = self.ir_builder.allocate_implicit_type()

        # Lower the value expression
        value_ref = self.lower_expr(expr.value)
        
        # Create a constant with the specified signal type and value
        if isinstance(value_ref, int):
            return self.ir_builder.const(signal_type, value_ref, expr)
        else:
            # For non-integer values, create a constant with value 0
            # and let the type system handle the rest
            return self.ir_builder.const(signal_type, 0, expr)

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
            return self.ir_builder.const(
                self.ir_builder.allocate_implicit_type(), 0, expr
            )

        memory_id = self.memory_refs[memory_name]
        data_ref = self.lower_expr(expr.value)

        # Write enable is always 1 (unconditional write)
        write_enable = self.ir_builder.const(
            self.ir_builder.allocate_implicit_type(), 1, expr
        )

        self.ir_builder.memory_write(memory_id, data_ref, write_enable, expr)

        # Write expressions return the written value
        return (
            data_ref
            if isinstance(data_ref, SignalRef)
            else self.ir_builder.const(
                self.ir_builder.allocate_implicit_type(), 0, expr
            )
        )

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
                self.diagnostics.warning(
                    f"Projection collapsing bundle to {target_type}", expr
                )
                # For now, just create a constant - proper summing would need more IR nodes
                return self.ir_builder.const(target_type, 0, expr)

        elif isinstance(source_ref, int):
            # Convert integer to signal of target type
            return self.ir_builder.const(target_type, source_ref, expr)

        else:
            self.diagnostics.error(
                f"Cannot project {type(source_ref)} to {target_type}", expr
            )
            return self.ir_builder.const(target_type, 0, expr)

    def lower_call_expr(self, expr: CallExpr) -> ValueRef:
        """Lower function call or special calls like place()."""
        if expr.name == "place":
            return self.lower_place_call(expr)
        else:
            # Try to inline simple function calls
            return self.lower_function_call_inline(expr)

    def lower_place_call(self, expr: CallExpr) -> SignalRef:
        """Lower place() call for entity placement."""
        if len(expr.args) < 3:
            self.diagnostics.error(
                "place() requires at least 3 arguments: (prototype, x, y)", expr
            )
            return self.ir_builder.const(
                self.ir_builder.allocate_implicit_type(), 0, expr
            )

        # Extract arguments
        prototype_expr = expr.args[0]
        x_expr = expr.args[1]
        y_expr = expr.args[2]

        # Get the prototype - it should be a string literal
        if isinstance(prototype_expr, StringLiteral):
            prototype = prototype_expr.value
        else:
            self.diagnostics.error(
                f"place() prototype must be a string literal, got {type(prototype_expr)}", prototype_expr
            )
            return self.ir_builder.const(
                self.ir_builder.allocate_implicit_type(), 0, expr
            )

        # Get coordinates - they can be SignalLiterals with NumberLiteral values or NumberLiterals
        def extract_coordinate(coord_expr):
            if isinstance(coord_expr, SignalLiteral) and isinstance(coord_expr.value, NumberLiteral):
                return coord_expr.value.value
            elif isinstance(coord_expr, NumberLiteral):
                return coord_expr.value
            else:
                # For non-constant coordinates, return the expression to be lowered
                return self.lower_expr(coord_expr)
        
        x_coord = extract_coordinate(x_expr)
        y_coord = extract_coordinate(y_expr)

        # Support both constant integers and variable coordinates
        # For variable coordinates, the actual placement will be resolved at emit time

        # Generate unique entity ID
        entity_id = f"entity_{self.ir_builder.next_id()}"

        # Place the entity (coordinates can be constants or signal references)
        self.ir_builder.place_entity(
            entity_id, prototype, x_coord, y_coord, source_ast=expr
        )

        # Return a dummy signal (entities don't produce signals directly)
        return self.ir_builder.const(self.ir_builder.allocate_implicit_type(), 0, expr)

    def lower_place_call_with_tracking(self, expr: CallExpr) -> tuple[str, ValueRef]:
        """Lower place() call and return both entity_id and value reference for tracking."""
        if len(expr.args) != 3:
            self.diagnostics.error(
                "place() requires exactly 3 arguments: prototype, x, y", expr
            )
            dummy_ref = self.ir_builder.const(
                self.ir_builder.allocate_implicit_type(), 0, expr
            )
            return "error_entity", dummy_ref

        # Extract arguments
        prototype_expr, x_expr, y_expr = expr.args

        # Get the prototype - it should be a string literal
        if isinstance(prototype_expr, StringLiteral):
            prototype = prototype_expr.value
        else:
            self.diagnostics.error(
                f"place() prototype must be a string literal, got {type(prototype_expr)}", prototype_expr
            )
            dummy_ref = self.ir_builder.const(
                self.ir_builder.allocate_implicit_type(), 0, expr
            )
            return "error_entity", dummy_ref

        # Get coordinates - they can be SignalLiterals with NumberLiteral values or NumberLiterals
        def extract_coordinate(coord_expr):
            if isinstance(coord_expr, SignalLiteral) and isinstance(coord_expr.value, NumberLiteral):
                return coord_expr.value.value
            elif isinstance(coord_expr, NumberLiteral):
                return coord_expr.value
            else:
                # For non-constant coordinates, return the expression to be lowered
                return self.lower_expr(coord_expr)
        
        x_coord = extract_coordinate(x_expr)
        y_coord = extract_coordinate(y_expr)

        # Generate unique entity ID
        entity_id = f"entity_{self.ir_builder.next_id()}"

        # Place the entity
        self.ir_builder.place_entity(
            entity_id, prototype, x_coord, y_coord, source_ast=expr
        )

        # Return both entity_id and dummy signal reference
        value_ref = self.ir_builder.const(
            self.ir_builder.allocate_implicit_type(), 0, expr
        )
        return entity_id, value_ref

    def lower_property_access(self, expr: PropertyAccess) -> ValueRef:
        """Lower property access for reading entity properties."""
        entity_name = expr.object_name
        prop_name = expr.property_name

        if entity_name in self.entity_refs:
            entity_id = self.entity_refs[entity_name]
            # Allocate a type for the property read result
            signal_type = self.ir_builder.allocate_implicit_type()

            # Create an IR operation for reading entity property
            read_op = IR_EntityPropRead(
                f"prop_read_{entity_id}_{prop_name}", signal_type, expr
            )
            read_op.entity_id = entity_id
            read_op.property_name = prop_name
            self.ir_builder.add_operation(read_op)

            # Return a signal reference pointing to this operation's output
            return SignalRef(read_op.node_id, signal_type)
        else:
            self.diagnostics.error(f"Undefined entity: {entity_name}", expr)
            return self.ir_builder.const(
                self.ir_builder.allocate_implicit_type(), 0, expr
            )

    def lower_property_access_expr(self, expr: PropertyAccessExpr) -> ValueRef:
        """Lower property access expression for reading entity properties."""
        entity_name = expr.object_name
        prop_name = expr.property_name

        if entity_name in self.entity_refs:
            entity_id = self.entity_refs[entity_name]
            # Allocate a type for the property read result
            signal_type = self.ir_builder.allocate_implicit_type()

            # Create an IR operation for reading entity property
            read_op = IR_EntityPropRead(
                f"prop_read_{entity_id}_{prop_name}", signal_type, expr
            )
            read_op.entity_id = entity_id
            read_op.property_name = prop_name
            self.ir_builder.add_operation(read_op)

            # Return a signal reference pointing to this operation's output
            return SignalRef(read_op.node_id, signal_type)
        else:
            self.diagnostics.error(f"Undefined entity: {entity_name}", expr)
            return self.ir_builder.const(
                self.ir_builder.allocate_implicit_type(), 0, expr
            )

    def lower_function_call_inline(self, expr: CallExpr) -> ValueRef:
        """Inline a function call by substituting parameters and lowering the body."""
        from dsl_compiler.src.dsl_ast import ReturnStmt
        
        # Look up the function definition
        func_symbol = self.semantic.current_scope.lookup(expr.name)
        if not func_symbol or func_symbol.symbol_type != "function" or not func_symbol.function_def:
            self.diagnostics.error(f"Cannot inline function: {expr.name}", expr)
            return self.ir_builder.const(self.ir_builder.allocate_implicit_type(), 0, expr)

        func_def = func_symbol.function_def
        
        # Check argument count
        if len(expr.args) != len(func_def.params):
            self.diagnostics.error(
                f"Function {expr.name} expects {len(func_def.params)} arguments, got {len(expr.args)}", expr
            )
            return self.ir_builder.const(self.ir_builder.allocate_implicit_type(), 0, expr)

        # Create parameter substitution map by evaluating arguments
        param_values = {}
        for param_name, arg_expr in zip(func_def.params, expr.args):
            param_values[param_name] = self.lower_expr(arg_expr)

        # Store old parameter values to restore later
        old_param_values = {}
        for param_name in func_def.params:
            if param_name in self.param_values:
                old_param_values[param_name] = self.param_values[param_name]

        # Set new parameter values
        self.param_values.update(param_values)

        try:
            # Store state to isolate function scope
            old_signal_refs = self.signal_refs.copy()
            old_entity_refs = self.entity_refs.copy()
            
            # Process all function body statements and collect return value
            return_value = None
            for stmt in func_def.body:
                if isinstance(stmt, ReturnStmt) and stmt.expr:
                    # Check if we're returning an entity variable
                    from dsl_compiler.src.dsl_ast import IdentifierExpr
                    if isinstance(stmt.expr, IdentifierExpr):
                        var_name = stmt.expr.name
                        if var_name in self.entity_refs:
                            # Mark this entity as the returned one
                            self.returned_entity_id = self.entity_refs[var_name]
                    
                    return_value = self.lower_expr(stmt.expr)
                    break
                else:
                    # Process other statements (declarations, assignments, etc.)
                    self.lower_statement(stmt)

            # Restore scoping but keep created entities for the caller
            created_entities = {}
            for name, entity_id in self.entity_refs.items():
                if name not in old_entity_refs:
                    created_entities[name] = entity_id
            
            self.signal_refs = old_signal_refs
            self.entity_refs = old_entity_refs
            
            # Re-add any newly created entities that weren't restored
            self.entity_refs.update(created_entities)

            # Return the collected value or a default
            if return_value is not None:
                return return_value
            else:
                return self.ir_builder.const(self.ir_builder.allocate_implicit_type(), 0, expr)

        finally:
            # Restore old parameter values
            for param_name in func_def.params:
                if param_name in old_param_values:
                    self.param_values[param_name] = old_param_values[param_name]
                else:
                    self.param_values.pop(param_name, None)


# =============================================================================
# Public API
# =============================================================================


def lower_program(
    program: Program, semantic_analyzer: SemanticAnalyzer
) -> tuple[List[IRNode], DiagnosticCollector, Dict[str, str]]:
    """Lower a semantic-analyzed program to IR."""
    lowerer = ASTLowerer(semantic_analyzer)
    ir_operations = lowerer.lower_program(program)
    return ir_operations, lowerer.diagnostics, lowerer.ir_builder.signal_type_map
