## Potential Improvements

### 1. Signal Resolution Complexity

**Location:** `dsl_compiler/src/layout/signal_resolver.py` and `dsl_compiler/src/layout/signal_analyzer.py`

**Observation:** There are multiple signal resolution methods across two classes:
- `SignalResolver.get_signal_name()`
- `SignalMaterializer.resolve_signal_name()`
- `SignalMaterializer.resolve_signal_type()`
- `SignalMaterializer._resolve_signal_identity()`

**Recommendation:** Consider consolidating signal resolution logic into a single class with clear responsibilities. The current split between SignalResolver and SignalMaterializer could be simplified.

**Priority:** Low (works correctly, just could be more elegant)

---

### 2. Entity Footprint Hardcoding

**Location:** `dsl_compiler/src/layout/entity_placer.py` lines 379-393

**Current Code:**
```python
def _get_entity_footprint(self, prototype: str) -> Tuple[int, int]:
    """Get entity footprint size (simplified version)."""
    FOOTPRINTS = {
        "train-stop": (2, 2),
        "roboport": (4, 4),
        # ... more hardcoded values
    }
    return FOOTPRINTS.get(prototype, (1, 1))
```

**Issue:** Hardcoded footprint table that will need updates as entities are added.

**Recommendation:** 
- Consider using Draftsman's entity data to get footprints dynamically
- Or move this to a centralized configuration file
- Or query Draftsman's `entity_prototypes` data

**Example:**
```python
def _get_entity_footprint(self, prototype: str) -> Tuple[int, int]:
    try:
        from draftsman.data import entities
        entity_data = entities.raw.get(prototype, {})
        collision_box = entity_data.get('collision_box', [[0, 0], [1, 1]])
        # Calculate footprint from collision_box
        width = max(1, int(collision_box[1][0] - collision_box[0][0]))
        height = max(1, int(collision_box[1][1] - collision_box[0][1]))
        return (width, height)
    except Exception:
        return (1, 1)  # Fallback
```

**Priority:** Medium (reduces maintenance burden)

---

### 3. ENTITY_ALIGNMENT Hardcoding

**Location:** `dsl_compiler/src/layout/entity_placer.py` lines 26-32

**Current Code:**
```python
ENTITY_ALIGNMENT = {
    'train-stop': 2,
    'roboport': 2,
    'substation': 2,
    # All other entities default to alignment=1
}
```

**Issue:** Similar to footprint issue - hardcoded and needs manual updates.

**Recommendation:** Alignment is always related to entity size. Could be computed from footprint:
```python
def _get_alignment(self, footprint: Tuple[int, int]) -> int:
    """Get alignment requirement from footprint."""
    # 2x2 or larger entities need alignment=2
    if max(footprint) >= 2:
        return 2
    return 1
```

**Priority:** Low (only affects a few entity types)

---

### 4. Duplicate Exception Handling Patterns

**Location:** Throughout codebase

**Observation:** Many identical exception handlers:
```python
except Exception as exc:  # pragma: no cover - draftsman errors
    self.diagnostics.warning(f"Could not...")
```

**Recommendation:** Create a helper decorator or context manager:
```python
@contextmanager
def handle_draftsman_errors(diagnostics, message):
    try:
        yield
    except Exception as exc:
        diagnostics.warning(f"{message}: {exc}")

# Usage:
with handle_draftsman_errors(self.diagnostics, "Failed to set property"):
    setattr(entity, prop_name, value)
```

**Priority:** Low (code smell, not a bug)

---

### 5. Magic Numbers in Connection Planning

**Location:** `dsl_compiler/src/layout/connection_planner.py`

**Observation:** Various magic numbers like:
- `max_radius=4` (line 308)
- `max_radius=6` (various places)
- `max_radius=12` (in reserve_near)
- `relay_cap = max(4096, base_segments * 8)` (line 320)

**Recommendation:** Extract to named constants at module level:
```python
# At top of file
MAX_ENTITY_SEARCH_RADIUS = 12
RELAY_SEARCH_RADIUS = 6
RELAY_INSERTION_RADIUS = 4
MAX_RELAY_CHAIN_LENGTH = 4096
RELAY_SAFETY_MULTIPLIER = 8
```

**Priority:** Low (readability improvement)

---

### 6. Property Writes Signal Type Hardcoded

**Location:** `dsl_compiler/src/emission/entity_emitter.py` line 153

**Current Code:**
```python
signal_dict = {
    "name": signal_name,
    "type": "virtual"  # Hardcoded assumption
}
```

**Issue:** Assumes all signals are "virtual" type, but they could be "item", "fluid", etc.

**Recommendation:** Determine signal type from signal_type_map or signal name:
```python
def _get_signal_type(self, signal_name: str) -> str:
    """Determine if signal is virtual, item, fluid, etc."""
    # Check if it's a known item/fluid in Factorio data
    try:
        from draftsman.data import items, fluids
        if signal_name in items.raw:
            return "item"
        if signal_name in fluids.raw:
            return "fluid"
    except Exception:
        pass
    return "virtual"
```

**Priority:** Medium (could cause issues with item/fluid signals)

---

### 7. Inconsistent Error Handling in Property Writes

**Location:** `dsl_compiler/src/emission/entity_emitter.py` lines 176-181

**Current Code:**
```python
else:
    # Other properties - try direct assignment
    try:
        setattr(entity, prop_name, prop_data.get("value"))
    except Exception:
        self.diagnostics.warning(...)
```

**Issue:** Silent failure with only a warning for unhandled property types.

**Recommendation:** Add explicit handling for known property types:
```python
KNOWN_PROPERTIES = {
    'enable': '_handle_enable_property',
    'recipe': '_handle_recipe_property',
    'filters': '_handle_filter_property',
    # etc.
}

def _apply_property_write(self, entity, prop_name, prop_data):
    handler = KNOWN_PROPERTIES.get(prop_name)
    if handler:
        getattr(self, handler)(entity, prop_data)
    else:
        # Generic fallback
        self._apply_generic_property(entity, prop_name, prop_data)
```

**Priority:** Medium (extensibility improvement)

---

### 8. Test File Naming Inconsistency

**Location:** `tests/` directory

**Observation:** Disabled test file uses `.legacy_disabled` extension

**Current:** `test_emit.py.legacy_disabled`

**Recommendation:** Either:
- Move to `tests/legacy/` directory
- Or delete entirely if not needed
- Or rename to `test_emit_legacy.py.disabled`

**Priority:** Low (organizational)

---

### 9. Entity Property Signal Conditions Incomplete

**Location:** `dsl_compiler/src/emission/entity_emitter.py` line 157-161

**Current Code:**
```python
if hasattr(entity, 'set_circuit_condition'):
    entity.set_circuit_condition(
        signal_dict,
        ">",
        0  # Hardcoded comparator and constant
    )
```

**Issue:** Hardcodes the circuit condition as `signal > 0`. The DSL might want different conditions.

**Recommendation:** Parse the actual condition from the IR. For now, the simple heuristic works, but in the future:
```python
# Store the comparison in IR_EntityPropWrite
# Then use it here:
comparator = prop_data.get('comparator', '>')
constant = prop_data.get('constant', 0)
entity.set_circuit_condition(signal_dict, comparator, constant)
```

**Priority:** Low (current behavior is reasonable default)

---

### 10. Memory Module Cleanup

**Location:** `dsl_compiler/src/layout/memory.py`

**Status:** This file still exists and is 124 lines

**Recommendation:** Review if this can be simplified or better integrated with the new EntityPlacer architecture. The memory system is complex but might benefit from refactoring.

**Priority:** Low (works correctly, future improvement)
