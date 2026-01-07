from __future__ import annotations

from .nodes import (
    IR_Arith,
    IR_Const,
    IR_Decider,
    IR_EntityPropWrite,
    IR_MemWrite,
    IRNode,
    SignalRef,
)

"""IR optimization passes for the Facto."""


class ConstantPropagationOptimizer:
    """Constant propagation and folding for IR operations.

    This pass folds operations with constant operands, BUT respects user-declared
    signals. User-declared signals MUST materialize as combinators, so operations
    that read from them are NOT folded.

    Only folds operations where ALL operands are anonymous (not user-declared).
    """

    def __init__(self) -> None:
        self.constants: dict[str, int] = {}  # node_id -> constant value
        self.replacements: dict[str, str] = {}  # old_id -> new_id
        self.dead_nodes: set[str] = set()  # nodes to remove

    def _is_user_declared_operand(self, value, const_map: dict[str, IR_Const]) -> bool:
        """Check if an operand references a user-declared constant."""
        if isinstance(value, SignalRef):
            node_id = value.source_id
            # Follow replacement chain
            while node_id in self.replacements:
                node_id = self.replacements[node_id]
            if node_id in const_map:
                return const_map[node_id].debug_metadata.get("user_declared", False)
        return False

    def optimize(self, ir_operations: list[IRNode]) -> list[IRNode]:
        """Perform constant propagation and folding."""

        # Build a map of all constants and their node IDs
        const_map = {}
        for op in ir_operations:
            if isinstance(op, IR_Const):
                const_map[op.node_id] = op

        # Iterate until no more folding possible (fixed point)
        changed = True
        while changed:
            changed = False

            for op in ir_operations:
                if op.node_id in self.dead_nodes:
                    continue

                if isinstance(op, IR_Arith):
                    # Skip folding if any operand is user-declared
                    if self._is_user_declared_operand(op.left, const_map):
                        continue
                    if self._is_user_declared_operand(op.right, const_map):
                        continue

                    left_val = self._get_const_value(op.left, const_map)
                    right_val = self._get_const_value(op.right, const_map)

                    if left_val is not None and right_val is not None:
                        # Both operands are constants, fold the operation
                        folded = self._fold_arithmetic(op.op, left_val, right_val)
                        if folded is not None:
                            # Create new constant or reuse existing
                            new_const = IR_Const(
                                op.node_id + "_folded", op.output_type, op.source_ast
                            )
                            new_const.value = folded

                            # Propagate suppress_materialization flag from original op
                            if op.debug_metadata.get("suppress_materialization"):
                                new_const.debug_metadata["suppress_materialization"] = True

                            const_map[new_const.node_id] = new_const

                            # Replace this operation with the constant
                            self.replacements[op.node_id] = new_const.node_id
                            self.dead_nodes.add(op.node_id)

                            # Mark operand constants as potentially dead
                            if isinstance(op.left, SignalRef):
                                self._maybe_mark_dead(
                                    op.left.source_id, const_map, ir_operations
                                )
                            if isinstance(op.right, SignalRef):
                                self._maybe_mark_dead(
                                    op.right.source_id, const_map, ir_operations
                                )

                            changed = True

                elif isinstance(op, IR_Decider):
                    # Handle multi-condition deciders (from condition folding)
                    if op.conditions:
                        # Multi-condition deciders use ValueRef operands stored in conditions
                        # Skip constant folding for now - these are more complex to evaluate
                        # and the inputs are often runtime signals, not compile-time constants
                        #
                        # TODO: Could fold if ALL conditions can be evaluated at compile time,
                        # but this is rare and the complexity isn't worth it for now.
                        continue

                    # Legacy single-condition mode
                    # Skip folding if any operand is user-declared
                    if self._is_user_declared_operand(op.left, const_map):
                        continue
                    if self._is_user_declared_operand(op.right, const_map):
                        continue

                    left_val = self._get_const_value(op.left, const_map)
                    right_val = self._get_const_value(op.right, const_map)

                    if left_val is not None and right_val is not None:
                        # Both operands are constants, fold the comparison
                        folded = self._fold_comparison(op.test_op, left_val, right_val)
                        if folded is not None:
                            # Decider outputs either output_value or 0
                            # If comparison is true and output_value is constant, output that
                            output_val = self._get_const_value(
                                op.output_value, const_map
                            )
                            if output_val is None:
                                output_val = 1  # Default output value

                            final_value = output_val if folded else 0

                            new_const = IR_Const(
                                op.node_id + "_folded", op.output_type, op.source_ast
                            )
                            new_const.value = final_value

                            # Propagate suppress_materialization flag from original op
                            if op.debug_metadata.get("suppress_materialization"):
                                new_const.debug_metadata["suppress_materialization"] = True

                            const_map[new_const.node_id] = new_const

                            self.replacements[op.node_id] = new_const.node_id
                            self.dead_nodes.add(op.node_id)

                            # Mark operand references as potentially dead
                            if isinstance(op.left, SignalRef):
                                self._maybe_mark_dead(
                                    op.left.source_id, const_map, ir_operations
                                )
                            if isinstance(op.right, SignalRef):
                                self._maybe_mark_dead(
                                    op.right.source_id, const_map, ir_operations
                                )

                            changed = True

        # Build final operation list
        result = []
        for op in ir_operations:
            if op.node_id in self.dead_nodes:
                continue
            result.append(op)

        # Add any new constants we created
        for node_id, replacement_id in self.replacements.items():
            if replacement_id in const_map and const_map[replacement_id] not in result:
                result.append(const_map[replacement_id])

        # Update all references
        if self.replacements:
            result = self._update_references(result)

        return result

    def _get_const_value(self, value, const_map: dict[str, IR_Const]) -> int | None:
        """Extract constant value from a ValueRef."""
        if isinstance(value, int):
            return value
        if isinstance(value, SignalRef):
            # Follow replacement chain
            node_id = value.source_id
            while node_id in self.replacements:
                node_id = self.replacements[node_id]

            if node_id in const_map:
                return const_map[node_id].value
        return None

    def _fold_arithmetic(self, op: str, left: int, right: int) -> int | None:
        """Fold an arithmetic operation on two constants."""
        try:
            if op == "+":
                return left + right
            elif op == "-":
                return left - right
            elif op == "*":
                return left * right
            elif op == "/":
                return left // right if right != 0 else None
            elif op == "%":
                return left % right if right != 0 else None
            elif op == "**" or op == "^":
                # Prevent overflow (^ is Factorio's power operator)
                if abs(left) > 1000 or abs(right) > 100:
                    return None
                if right < 0:
                    return 0
                return left**right
            elif op == "<<":
                if right < 0 or right >= 32:
                    return 0
                return (left << right) & 0xFFFFFFFF
            elif op == ">>":
                if right < 0 or right >= 32:
                    return 0
                return left >> right
            elif op == "&" or op == "AND":
                return left & right
            elif op == "|" or op == "OR":
                return left | right
            elif op == "XOR":
                return left ^ right
        except (OverflowError, ValueError):
            return None
        return None

    def _fold_comparison(self, op: str, left: int, right: int) -> bool | None:
        """Evaluate a comparison operation on two constants.

        Returns True/False for the comparison result, or None if the operator
        is not recognized.
        """
        if op == "==" or op == "=":
            return left == right
        elif op == "!=" or op == "≠":
            return left != right
        elif op == "<":
            return left < right
        elif op == "<=":
            return left <= right
        elif op == ">":
            return left > right
        elif op == ">=":
            return left >= right
        return None

    def _maybe_mark_dead(
        self, node_id: str, const_map: dict[str, IR_Const], all_ops: list[IRNode]
    ) -> None:
        """Mark a constant as dead if it has no other consumers.

        NEVER marks user-declared constants as dead - they must materialize.
        """
        if node_id not in const_map:
            return

        # Never mark user-declared constants as dead
        if const_map[node_id].debug_metadata.get("user_declared", False):
            return

        # Check if this constant is used anywhere else
        for op in all_ops:
            if op.node_id == node_id or op.node_id in self.dead_nodes:
                continue

            # Check all operands
            if isinstance(op, IR_Arith):
                if self._references_node(op.left, node_id) or self._references_node(
                    op.right, node_id
                ):
                    return
            elif isinstance(op, IR_Decider):
                if (
                    self._references_node(op.left, node_id)
                    or self._references_node(op.right, node_id)
                    or self._references_node(op.output_value, node_id)
                ):
                    return

        # No consumers found, mark as dead
        self.dead_nodes.add(node_id)

    def _references_node(self, value, target_id: str) -> bool:
        """Check if a value references a specific node."""
        if isinstance(value, SignalRef):
            return value.source_id == target_id
        return False

    def _update_references(self, operations: list[IRNode]) -> list[IRNode]:
        """Update all SignalRef references to use replacement nodes."""
        for op in operations:
            if isinstance(op, IR_Arith):
                op.left = self._update_value(op.left)
                op.right = self._update_value(op.right)
            elif isinstance(op, IR_Decider):
                op.left = self._update_value(op.left)
                op.right = self._update_value(op.right)
                op.output_value = self._update_value(op.output_value)
            elif isinstance(op, IR_MemWrite):
                op.data_signal = self._update_value(op.data_signal)
                op.write_enable = self._update_value(op.write_enable)

        return operations

    def _update_value(self, value):
        """Update a ValueRef to use replacement node if available."""
        if isinstance(value, SignalRef):
            canonical = self.replacements.get(value.source_id)
            if canonical:
                return SignalRef(
                    value.signal_type,
                    canonical,
                    debug_label=value.debug_label,
                    source_ast=value.source_ast,
                    metadata=value.debug_metadata,
                )
        return value


class CSEOptimizer:
    """Common subexpression elimination for IR nodes."""

    def __init__(self) -> None:
        self.expr_cache: dict[str, str] = {}
        self.replacements: dict[str, str] = {}

    def optimize(self, ir_operations: list[IRNode]) -> list[IRNode]:
        """Eliminate redundant arithmetic and decider operations."""

        optimized: list[IRNode] = []

        for op in ir_operations:
            if isinstance(op, (IR_Arith, IR_Decider)):
                key = self._make_key(op)
                if key and key in self.expr_cache:
                    self.replacements[op.node_id] = self.expr_cache[key]
                    continue
                if key:
                    self.expr_cache[key] = op.node_id

            optimized.append(op)

        if self.replacements:
            optimized = self._update_references(optimized)

        return optimized

    def _make_key(self, op: IRNode) -> str:
        if isinstance(op, IR_Arith):
            left_key = self._value_key(op.left)
            right_key = self._value_key(op.right)
            return f"arith:{op.op}:{left_key}:{right_key}:{op.output_type}"

        if isinstance(op, IR_Decider):
            # Multi-condition mode
            if op.conditions:
                cond_keys = []
                for c in op.conditions:
                    # Handle both ValueRef-based and string-based operands
                    if c.first_operand is not None:
                        first_key = self._value_key(c.first_operand)
                    elif c.first_signal:
                        first_key = f"str:{c.first_signal}"
                    else:
                        first_key = f"int:{c.first_constant}"

                    if c.second_operand is not None:
                        second_key = self._value_key(c.second_operand)
                    elif c.second_signal:
                        second_key = f"str:{c.second_signal}"
                    else:
                        second_key = f"int:{c.second_constant}"

                    cond_keys.append(f"{c.comparator}:{first_key}:{second_key}:{c.compare_type}")

                output_key = self._value_key(op.output_value)
                return f"decider_multi:{':'.join(cond_keys)}:{output_key}:{op.output_type}"

            # Legacy single-condition mode
            left_key = self._value_key(op.left)
            right_key = self._value_key(op.right)
            output_key = self._value_key(op.output_value)
            return (
                "decider:"
                f"{op.test_op}:"
                f"{left_key}:"
                f"{right_key}:"
                f"{output_key}:"
                f"{op.output_type}"
            )

        return ""

    def _value_key(self, value) -> str:
        if isinstance(value, SignalRef):
            source_id = self.replacements.get(value.source_id, value.source_id)
            return f"sig:{source_id}:{value.signal_type}"

        if isinstance(value, int):
            return f"int:{value}"

        if isinstance(value, str):
            return f"str:{value}"

        return repr(value)

    def _update_references(self, operations: list[IRNode]) -> list[IRNode]:
        for op in operations:
            if isinstance(op, IR_Arith):
                op.left = self._update_value(op.left)
                op.right = self._update_value(op.right)
            elif isinstance(op, IR_Decider):
                # Update legacy single-condition fields
                op.left = self._update_value(op.left)
                op.right = self._update_value(op.right)
                op.output_value = self._update_value(op.output_value)
                # Update multi-condition operands
                for cond in op.conditions:
                    if cond.first_operand is not None:
                        cond.first_operand = self._update_value(cond.first_operand)
                    if cond.second_operand is not None:
                        cond.second_operand = self._update_value(cond.second_operand)
            elif isinstance(op, IR_MemWrite):
                op.data_signal = self._update_value(op.data_signal)
                op.write_enable = self._update_value(op.write_enable)
            elif isinstance(op, IR_EntityPropWrite):
                op.value = self._update_value(op.value)

        return operations

    def _update_value(self, value):
        if isinstance(value, SignalRef):
            canonical = self.replacements.get(value.source_id)
            if canonical:
                return SignalRef(
                    value.signal_type,
                    canonical,
                    debug_label=value.debug_label,
                    source_ast=value.source_ast,
                    metadata=value.debug_metadata,
                )
        return value


__all__ = ["CSEOptimizer", "ConstantPropagationOptimizer"]
