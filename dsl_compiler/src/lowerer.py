# lowerer.py
"""
AST to IR lowering pass for the Factorio Circuit DSL.

This module converts semantic-analyzed AST nodes into IR operations
following the lowering rules specified in the compiler specification.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union, Any

from dsl_compiler.src.dsl_ast import *
from dsl_compiler.src.ir import *
from dsl_compiler.src.semantic import (
    SemanticAnalyzer,
    SignalValue,
    IntValue,
    DiagnosticCollector,
    render_source_location,
)

from draftsman.data import signals as signal_data


class ASTLowerer:
    """Converts semantic-analyzed AST to IR."""

    def __init__(self, semantic_analyzer: SemanticAnalyzer):
        self.semantic = semantic_analyzer
        self.ir_builder = IRBuilder()
        self.diagnostics = DiagnosticCollector()

        # Symbol tables for IR references
        self.signal_refs: Dict[str, SignalRef] = {}  # variable_name -> SignalRef
        self.memory_refs: Dict[str, str] = {}  # memory_name -> memory_id
        self.entity_refs: Dict[str, str] = {}  # entity_name -> entity_id
        self.param_values: Dict[
            str, ValueRef
        ] = {}  # parameter_name -> value during function inlining

        # Copy signal type mapping from semantic analyzer
        self.ir_builder.signal_type_map = self.semantic.signal_type_map.copy()

        # Hidden structures for compiler-generated helpers
        self._once_counter = 0

    # ------------------------------------------------------------------
    # Debug metadata helpers
    # ------------------------------------------------------------------

    def _annotate_signal_ref(self, name: str, ref: ValueRef, node: ASTNode) -> None:
        """Attach debug metadata for a lowered signal reference."""

        if not isinstance(ref, SignalRef):
            return

        location = render_source_location(node, getattr(node, "source_file", None))
        metadata = {"name": name}
        if location:
            metadata["location"] = location

        semantic_info = None
        if hasattr(self.semantic, "get_signal_debug_payload"):
            semantic_info = self.semantic.get_signal_debug_payload(name)

        source_ast = node
        if semantic_info:
            semantic_payload = semantic_info.as_dict()
            for key, value in semantic_payload.items():
                if value is None:
                    continue
                if key == "source_ast":
                    source_ast = value or source_ast
                    continue
                metadata.setdefault(key, value)

            # Provide canonical factorio signal name alias for emitters
            factorio_signal = semantic_payload.get("factorio_signal")
            if factorio_signal:
                metadata.setdefault("factorio_signal", factorio_signal)
                metadata.setdefault("resolved_signal", factorio_signal)

            signal_key = semantic_payload.get("signal_key")
            if signal_key:
                metadata.setdefault("signal_key", signal_key)

        self.ir_builder.annotate_signal(
            ref, label=name, source_ast=source_ast, metadata=metadata
        )

    def _infer_signal_category(self, signal_type: Optional[str]) -> str:
        """Infer the Factorio signal category for the given identifier."""
        if not signal_type:
            return "virtual"

        if signal_data is not None:
            if signal_type in signal_data.type_of:
                types = signal_data.type_of.get(signal_type, [])
                if types:
                    return types[0]
            if signal_type in signal_data.raw:
                info = signal_data.raw.get(signal_type, {})
                prototype_type = info.get("type")
                # Draftsman uses "virtual-signal" for prototype metadata, but
                # the registration API expects the simplified "virtual" label.
                if prototype_type == "virtual-signal":
                    return "virtual"
                if prototype_type in {
                    "item",
                    "fluid",
                    "recipe",
                    "entity",
                    "space-location",
                    "asteroid-chunk",
                    "quality",
                    "virtual",
                }:
                    return prototype_type

        mapped = self.ir_builder.signal_type_map.get(signal_type)
        if isinstance(mapped, dict):
            return mapped.get("type", "virtual")
        if isinstance(mapped, str):
            if signal_data is not None and mapped in signal_data.raw:
                prototype_type = signal_data.raw[mapped].get("type", "virtual")
                # Normalize Draftsman's "virtual-signal" prototype type to the
                # canonical "virtual" category before registering custom names.
                return (
                    "virtual" if prototype_type == "virtual-signal" else prototype_type
                )
            if mapped.startswith("signal-"):
                return "virtual"

        if signal_type.startswith("__v"):
            return "virtual"

        return "virtual"

    def _ensure_signal_registered(
        self, signal_type: Optional[str], source_signal_type: Optional[str] = None
    ) -> None:
        """Ensure that a signal identifier is known to the emitter."""
        if not signal_type:
            return

        if signal_data is not None and signal_type in signal_data.raw:
            return

        if signal_type in self.ir_builder.signal_type_map:
            return

        category = (
            self._infer_signal_category(source_signal_type)
            if source_signal_type
            else None
        )
        if not category:
            category = self._infer_signal_category(signal_type)

        self.ir_builder.signal_type_map[signal_type] = {
            "name": signal_type,
            "type": category or "virtual",
        }

    def _memory_signal_type(self, memory_name: str) -> Optional[str]:
        """Look up the declared signal type for a memory symbol."""

        mem_info = getattr(self.semantic, "memory_types", {}).get(memory_name)
        if mem_info and getattr(mem_info, "signal_type", None):
            return mem_info.signal_type

        symbol = self.semantic.symbol_table.lookup(memory_name)
        if symbol and isinstance(symbol.value_type, SignalValue):
            signal_info = symbol.value_type.signal_type
            if signal_info and getattr(signal_info, "name", None):
                return signal_info.name

        return None

    def _coerce_to_signal_type(
        self, value_ref: ValueRef, signal_type: str, node: ASTNode
    ) -> SignalRef:
        """Ensure a value is represented as a SignalRef of the desired type."""

        self._ensure_signal_registered(signal_type)

        if isinstance(value_ref, SignalRef):
            if value_ref.signal_type == signal_type:
                return value_ref

            # Arithmetic passthrough preserves value while retargeting type.
            return self.ir_builder.arithmetic(
                "+", value_ref, 0, signal_type, node
            )

        if isinstance(value_ref, int):
            return self.ir_builder.const(signal_type, value_ref, node)

        if hasattr(value_ref, "signal_type") and getattr(
            value_ref, "signal_type", None
        ) == signal_type:
            return value_ref  # type: ignore[return-value]

        self.diagnostics.error(
            f"Cannot convert value of type '{type(value_ref).__name__}' to signal '{signal_type}' for memory write.",
            node,
        )
        return self.ir_builder.const(signal_type, 0, node)

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
        # Special handling for place() calls to track entities
        if isinstance(stmt.value, CallExpr) and stmt.value.name == "place":
            entity_id, value_ref = self.lower_place_call_with_tracking(stmt.value)
            self.entity_refs[stmt.name] = entity_id
            self.signal_refs[stmt.name] = value_ref
            self._annotate_signal_ref(stmt.name, value_ref, stmt)
            return

        # Special handling for function calls that return entities
        if isinstance(stmt.value, CallExpr):
            # Clear any returned entity from previous call
            self.returned_entity_id = None
            value_ref = self.lower_expr(stmt.value)

            # If the function call returned an entity, track it
            if (
                hasattr(self, "returned_entity_id")
                and self.returned_entity_id is not None
            ):
                self.entity_refs[stmt.name] = self.returned_entity_id
                self.returned_entity_id = None

            self.signal_refs[stmt.name] = value_ref
            self._annotate_signal_ref(stmt.name, value_ref, stmt)
            return

        value_ref = self.lower_expr(stmt.value)

        # Store the reference for later use
        if isinstance(value_ref, SignalRef):
            self.signal_refs[stmt.name] = value_ref
            self._annotate_signal_ref(stmt.name, value_ref, stmt)
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
            self._annotate_signal_ref(stmt.name, const_ref, stmt)

    def lower_assign_stmt(self, stmt: AssignStmt):
        """Lower assignment statement: target = expr;"""
        if isinstance(stmt.target, Identifier):
            # Special handling for place() calls to track entities
            if isinstance(stmt.value, CallExpr) and stmt.value.name == "place":
                entity_id, value_ref = self.lower_place_call_with_tracking(stmt.value)
                self.entity_refs[stmt.target.name] = entity_id
                self.signal_refs[stmt.target.name] = value_ref
                self._annotate_signal_ref(stmt.target.name, value_ref, stmt)
                return

            # Special handling for function calls that return entities
            if isinstance(stmt.value, CallExpr):
                # Clear any returned entity from previous call
                self.returned_entity_id = None
                value_ref = self.lower_expr(stmt.value)

                # If the function call returned an entity, track it
                if (
                    hasattr(self, "returned_entity_id")
                    and self.returned_entity_id is not None
                ):
                    self.entity_refs[stmt.target.name] = self.returned_entity_id
                    self.returned_entity_id = None

                self.signal_refs[stmt.target.name] = value_ref
                self._annotate_signal_ref(stmt.target.name, value_ref, stmt)
                return

        value_ref = self.lower_expr(stmt.value)

        if isinstance(stmt.target, Identifier):
            # Simple variable assignment
            if isinstance(value_ref, SignalRef):
                self.signal_refs[stmt.target.name] = value_ref
                self._annotate_signal_ref(stmt.target.name, value_ref, stmt)
            elif isinstance(value_ref, int):
                symbol = self.semantic.symbol_table.lookup(stmt.target.name)
                if symbol and isinstance(symbol.value_type, SignalValue):
                    signal_type = symbol.value_type.signal_type.name
                else:
                    signal_type = self.ir_builder.allocate_implicit_type()

                const_ref = self.ir_builder.const(signal_type, value_ref, stmt)
                self.signal_refs[stmt.target.name] = const_ref
                self._annotate_signal_ref(stmt.target.name, const_ref, stmt)

        elif isinstance(stmt.target, PropertyAccess):
            # Entity property assignment
            entity_name = stmt.target.object_name
            prop_name = stmt.target.property_name
            if entity_name in self.entity_refs:
                entity_id = self.entity_refs[entity_name]
                prop_write_op = IR_EntityPropWrite(entity_id, prop_name, value_ref)
                self.ir_builder.add_operation(prop_write_op)

    def lower_mem_decl(self, stmt: MemDecl):
        """Lower memory declaration: Memory name: 'signal';"""
        memory_id = f"mem_{stmt.name}"
        self.memory_refs[stmt.name] = memory_id

        signal_type: Optional[str] = None
        mem_info = getattr(self.semantic, "memory_types", {}).get(stmt.name)

        if mem_info:
            signal_type = mem_info.signal_type
        else:
            symbol = self.semantic.symbol_table.lookup(stmt.name)
            if symbol and isinstance(symbol.value_type, SignalValue):
                signal_type = symbol.value_type.signal_type.name

        if signal_type is None:
            self.diagnostics.error(
                f"Memory '{stmt.name}' must have an explicit signal type.",
                stmt,
            )
            signal_type = "signal-0"

        self._ensure_signal_registered(signal_type)

        self.ir_builder.memory_create(memory_id, signal_type, stmt)

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
            f"Import statement found in AST - file not found during preprocessing: {stmt.path}",
            stmt,
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
            ProjectionExpr: self.lower_projection_expr,
            CallExpr: self.lower_call_expr,
            PropertyAccess: self.lower_property_access,
            PropertyAccessExpr: self.lower_property_access_expr,
            SignalLiteral: self.lower_signal_literal,
            DictLiteral: self.lower_dict_literal,
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
        else:
            self.diagnostics.error(f"Undefined identifier: {name}", expr)
            return self.ir_builder.const(
                self.ir_builder.allocate_implicit_type(), 0, expr
            )

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

        left_signal_type = (
            left_type.signal_type.name if isinstance(left_type, SignalValue) else None
        )
        self._ensure_signal_registered(output_type, left_signal_type)

        # Handle different operand combinations
        if expr.op in ["+", "-", "*", "/", "%"]:
            return self.ir_builder.arithmetic(
                expr.op, left_ref, right_ref, output_type, expr
            )
        elif expr.op in ["==", "!=", "<", "<=", ">", ">=", "&&", "||"]:
            # This should be handled by comparison lowering
            return self.lower_comparison_op(expr, left_ref, right_ref, output_type)
        else:
            self.diagnostics.error(f"Unknown binary operator: {expr.op}", expr)
            return self.ir_builder.const(output_type, 0, expr)

    def lower_comparison_op(
        self,
        expr: BinaryOp,
        left_ref: ValueRef,
        right_ref: ValueRef,
        output_type: Optional[str] = None,
    ) -> ValueRef:
        """Lower comparison operations to decider combinators."""
        # Use the semantically inferred output type when available
        if not output_type:
            result_type = self.semantic.get_expr_type(expr)
            if isinstance(result_type, SignalValue):
                output_type = result_type.signal_type.name
            else:
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

        operand_signal_type = (
            result_type.signal_type.name
            if isinstance(result_type, SignalValue)
            else None
        )
        self._ensure_signal_registered(output_type, operand_signal_type)

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
        # For explicit typed literals, always emit IR_Const for the declared type (item signal)
        if expr.signal_type is not None:
            signal_name = expr.signal_type
            output_type = expr.signal_type
            self._ensure_signal_registered(signal_name)
            value_ref = self.lower_expr(expr.value)
            # Emit IR_Const for the item signal with the correct value
            if isinstance(value_ref, int):
                ref = self.ir_builder.const(output_type, value_ref, expr)
            else:
                ref = self.ir_builder.const(output_type, 0, expr)
            ref.signal_type = signal_name
            ref.output_type = signal_name
            return ref
        else:
            # Fallback: emit IR_Const for virtual signal
            signal_name = self.ir_builder.allocate_implicit_type()
            output_type = signal_name
            self._ensure_signal_registered(signal_name)
            value_ref = self.lower_expr(expr.value)
            if isinstance(value_ref, int):
                ref = self.ir_builder.const(output_type, value_ref, expr)
            else:
                ref = self.ir_builder.const(output_type, 0, expr)
            ref.signal_type = signal_name
            ref.output_type = signal_name
            return ref

    def lower_dict_literal(self, expr: DictLiteral) -> Dict[str, Any]:
        """Lower dictionary literal to static property map."""
        properties: Dict[str, Any] = {}
        for key, value_expr in expr.entries.items():
            if isinstance(value_expr, NumberLiteral):
                properties[key] = value_expr.value
            elif isinstance(value_expr, StringLiteral):
                properties[key] = value_expr.value
            elif isinstance(value_expr, SignalLiteral):
                inner_value = value_expr.value
                if isinstance(inner_value, NumberLiteral):
                    properties[key] = inner_value.value
                elif isinstance(inner_value, StringLiteral):
                    properties[key] = inner_value.value
                else:
                    lowered = self.lower_expr(inner_value)
                    if isinstance(lowered, (int, str)):
                        properties[key] = lowered
                    else:
                        self.diagnostics.error(
                            f"Unsupported value for property '{key}' in place() call",
                            value_expr,
                        )
            else:
                lowered = self.lower_expr(value_expr)
                if isinstance(lowered, (int, str)):
                    properties[key] = lowered
                else:
                    self.diagnostics.error(
                        f"Unsupported value for property '{key}' in place() call",
                        value_expr,
                    )
        return properties

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

        self._ensure_signal_registered(signal_type)

        return self.ir_builder.memory_read(memory_id, signal_type, expr)

    def lower_write_expr(self, expr: WriteExpr) -> SignalRef:
        """Lower memory write: write(value, memory, when=...)."""
        memory_name = expr.memory_name

        if memory_name not in self.memory_refs:
            self.diagnostics.error(f"Undefined memory: {memory_name}", expr)
            return self.ir_builder.const(
                self.ir_builder.allocate_implicit_type(), 0, expr
            )

        memory_id = self.memory_refs[memory_name]
        data_ref = self.lower_expr(expr.value)

        expected_signal_type = self._memory_signal_type(memory_name)
        if expected_signal_type is None:
            self.diagnostics.error(
                f"Memory '{memory_name}' does not have a resolved signal type during lowering.",
                expr,
            )
            expected_signal_type = self.ir_builder.allocate_implicit_type()

        coerced_data_ref = self._coerce_to_signal_type(
            data_ref, expected_signal_type, expr
        )

        if getattr(expr, "when_once", False):
            write_enable = self._lower_once_enable(expr)
        elif expr.when is not None:
            write_enable = self.lower_expr(expr.when)
        else:
            self._ensure_signal_registered("signal-W")
            write_enable = self.ir_builder.const("signal-W", 1, expr)

        self.ir_builder.memory_write(memory_id, coerced_data_ref, write_enable, expr)

        # Write expressions return the written value
        return coerced_data_ref

    def _lower_once_enable(self, expr: WriteExpr) -> SignalRef:
        """Generate IR for the when=once guard."""
        self._once_counter += 1
        flag_name = f"__once_flag_{self._once_counter}"
        flag_memory_id = f"mem_{flag_name}"

        flag_signal_type = "signal-W"
        self._ensure_signal_registered(flag_signal_type)

        self.ir_builder.memory_create(flag_memory_id, flag_signal_type, expr)

        flag_read = self.ir_builder.memory_read(
            flag_memory_id, flag_signal_type, expr
        )
        condition = self.ir_builder.decider(
            "==", flag_read, 0, 1, flag_signal_type, expr
        )

        one_const = self.ir_builder.const(flag_signal_type, 1, expr)
        self.ir_builder.memory_write(flag_memory_id, one_const, condition, expr)

        return condition

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
                self._ensure_signal_registered(target_type, source_ref.signal_type)
                return self.ir_builder.arithmetic("+", source_ref, 0, target_type, expr)

        elif isinstance(source_ref, int):
            # Convert integer to signal of target type
            self._ensure_signal_registered(target_type)
            return self.ir_builder.const(target_type, source_ref, expr)

        else:
            self.diagnostics.error(
                f"Cannot project {type(source_ref)} to {target_type}", expr
            )
            self._ensure_signal_registered(target_type)
            return self.ir_builder.const(target_type, 0, expr)

    def lower_call_expr(self, expr: CallExpr) -> ValueRef:
        """Lower function call or special calls like place()/memory()."""
        if expr.name == "place":
            return self.lower_place_call(expr)
        if expr.name == "memory":
            return self.lower_memory_call(expr)
        # Try to inline simple function calls
        return self.lower_function_call_inline(expr)

    def _extract_coordinate(self, coord_expr: Expr) -> Union[int, ValueRef]:
        """Helper to lower place() coordinate argument."""
        if isinstance(coord_expr, SignalLiteral) and isinstance(
            coord_expr.value, NumberLiteral
        ):
            return coord_expr.value.value
        if isinstance(coord_expr, NumberLiteral):
            return coord_expr.value
        return self.lower_expr(coord_expr)

    def _lower_place_core(self, expr: CallExpr) -> tuple[str, SignalRef]:
        if len(expr.args) < 3:
            self.diagnostics.error(
                "place() requires at least 3 arguments: (prototype, x, y)", expr
            )
            dummy = self.ir_builder.const(
                self.ir_builder.allocate_implicit_type(), 0, expr
            )
            return "error_entity", dummy

        prototype_expr = expr.args[0]
        x_expr = expr.args[1]
        y_expr = expr.args[2]

        if isinstance(prototype_expr, StringLiteral):
            prototype = prototype_expr.value
        else:
            self.diagnostics.error(
                "place() prototype must be a string literal",
                prototype_expr,
            )
            dummy = self.ir_builder.const(
                self.ir_builder.allocate_implicit_type(), 0, expr
            )
            return "error_entity", dummy

        x_coord = self._extract_coordinate(x_expr)
        y_coord = self._extract_coordinate(y_expr)

        properties: Optional[Dict[str, Any]] = None
        if len(expr.args) >= 4:
            prop_expr = expr.args[3]
            if isinstance(prop_expr, DictLiteral):
                properties = self.lower_dict_literal(prop_expr)
            else:
                self.diagnostics.error(
                    "place() properties argument must be a dictionary literal",
                    prop_expr,
                )

        entity_id = f"entity_{self.ir_builder.next_id()}"
        self.ir_builder.place_entity(
            entity_id,
            prototype,
            x_coord,
            y_coord,
            properties=properties,
            source_ast=expr,
        )

        result_ref = self.ir_builder.const(
            self.ir_builder.allocate_implicit_type(), 0, expr
        )
        return entity_id, result_ref

    def lower_place_call(self, expr: CallExpr) -> SignalRef:
        """Lower place() call for entity placement."""
        _, value_ref = self._lower_place_core(expr)
        return value_ref

    def lower_place_call_with_tracking(self, expr: CallExpr) -> tuple[str, ValueRef]:
        """Lower place() call and return both entity_id and value reference for tracking."""
        return self._lower_place_core(expr)

    def lower_memory_call(self, expr: CallExpr) -> ValueRef:
        """Lower memory() helper used inside declarations."""
        if not expr.args:
            self.diagnostics.error("memory() requires an initial value", expr)
            return self.ir_builder.const(
                self.ir_builder.allocate_implicit_type(), 0, expr
            )

        initial_ref = self.lower_expr(expr.args[0])

        return initial_ref

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
            return SignalRef(signal_type, read_op.node_id)
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
            return SignalRef(signal_type, read_op.node_id)
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
        if (
            not func_symbol
            or func_symbol.symbol_type != "function"
            or not func_symbol.function_def
        ):
            self.diagnostics.error(f"Cannot inline function: {expr.name}", expr)
            return self.ir_builder.const(
                self.ir_builder.allocate_implicit_type(), 0, expr
            )

        func_def = func_symbol.function_def

        # Check argument count
        if len(expr.args) != len(func_def.params):
            self.diagnostics.error(
                f"Function {expr.name} expects {len(func_def.params)} arguments, got {len(expr.args)}",
                expr,
            )
            return self.ir_builder.const(
                self.ir_builder.allocate_implicit_type(), 0, expr
            )

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
                return self.ir_builder.const(
                    self.ir_builder.allocate_implicit_type(), 0, expr
                )

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
