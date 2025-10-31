# Factorio Circuit DSL: Layout Module Extraction Refactoring Guide

## Overview

This guide provides a methodical, step-by-step approach to extracting layout and planning logic from the emission module into a new dedicated layout module. Each step is designed to maintain a working, testable state.

## Current Architecture Analysis

**Current Flow:**
```
Parse → Semantic → Lowering → Emission (interleaved layout/emission) → Blueprint
```

**Target Flow:**
```
Parse → Semantic → Lowering → Layout Planning → Emission (thin wrapper) → Blueprint
```

**Files to Move/Delete from emission module:**
- `layout.py` → move to layout module
- `connection_builder.py` → move to layout module  
- `signal_resolver.py` → move to layout module
- `wiring.py` → move to layout module
- `signals.py` → move to layout module (partially)

**New Module Structure:**
```
dsl_compiler/src/layout/
  __init__.py
  layout_engine.py      # from emission/layout.py
  planner.py            # new: orchestrates layout planning
  signal_analyzer.py    # from emission/signals.py
  connection_planner.py # from emission/connection_builder.py
  wire_router.py        # from emission/wiring.py
  power_planner.py      # power pole logic extracted from emitter.py
  layout_plan.py        # new: LayoutPlan data structure
```

---

## Phase 1: Create Layout Module Structure (No Behavior Change)

**Objective:** Set up the new module and move layout engine without changing any behavior.

### Step 1.1: Create layout module directory structure ✅ Completed

```bash
mkdir -p dsl_compiler/src/layout
touch dsl_compiler/src/layout/__init__.py
```

### Step 1.2: Copy layout.py to new module ✅ Completed

**Action:**
1. Copy `dsl_compiler/src/emission/layout.py` to `dsl_compiler/src/layout/layout_engine.py`
2. Update the new file's imports (it currently has no imports, so no changes needed)
3. Update `dsl_compiler/src/layout/__init__.py`:

```python
"""Layout planning module for the Factorio Circuit DSL."""

from .layout_engine import LayoutEngine

__all__ = ["LayoutEngine"]
```

### Step 1.3: Update emission module imports ✅ Completed

**Files to update:**
- `dsl_compiler/src/emission/__init__.py` - change import
- `dsl_compiler/src/emission/emitter.py` - change import
- `dsl_compiler/src/emission/connection_builder.py` - change import
- `dsl_compiler/src/emission/memory.py` - change import

**Changes:**
```python
# OLD:
from .layout import LayoutEngine

# NEW:
from dsl_compiler.src.layout import LayoutEngine
```

### Step 1.4: Delete old layout.py ✅ Completed

```bash
rm dsl_compiler/src/emission/layout.py
```

### Step 1.5: Run tests ✅ Completed (2025-10-30, 116 passed)

```bash
python -m pytest tests/ -v
```

**Expected:** All tests pass. No behavior change.

---

## Phase 2: Create LayoutPlan Data Structure

**Objective:** Define the data structure that will hold all physical layout decisions.

### Step 2.1: Create layout_plan.py ✅ Completed

Create `dsl_compiler/src/layout/layout_plan.py`:

```python
"""Data structures for physical layout planning."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any

from dsl_compiler.src.ir import SignalRef


@dataclass
class EntityPlacement:
    """Physical placement of an IR entity."""
    ir_node_id: str
    entity_type: str  # draftsman entity type
    position: Tuple[int, int]
    properties: Dict[str, Any] = field(default_factory=dict)
    role: Optional[str] = None
    zone: Optional[str] = None


@dataclass
class WireConnection:
    """A physical wire connection."""
    source_entity_id: str
    sink_entity_id: str
    signal_name: str
    wire_color: str  # "red" or "green"
    source_side: Optional[str] = None  # "input" or "output" for dual-sided
    sink_side: Optional[str] = None


@dataclass
class PowerPolePlacement:
    """Physical power pole placement."""
    pole_id: str
    pole_type: str
    position: Tuple[int, int]


@dataclass
class SignalMaterialization:
    """Decision about whether/how to materialize a signal."""
    signal_id: str
    should_materialize: bool
    resolved_signal_name: Optional[str] = None
    resolved_signal_type: Optional[str] = None
    is_inlinable_constant: bool = False
    constant_value: Optional[int] = None


@dataclass
class LayoutPlan:
    """Complete physical layout plan for blueprint emission."""
    
    # Entity placements
    entity_placements: Dict[str, EntityPlacement] = field(default_factory=dict)
    
    # Wire connections
    wire_connections: List[WireConnection] = field(default_factory=list)
    
    # Power infrastructure
    power_poles: List[PowerPolePlacement] = field(default_factory=list)
    
    # Signal decisions
    signal_materializations: Dict[str, SignalMaterialization] = field(default_factory=dict)
    
    # Signal connectivity graph (source -> sinks)
    signal_graph: Dict[str, List[str]] = field(default_factory=dict)
    
    # Metadata
    blueprint_label: str = "DSL Generated"
    blueprint_description: str = ""
    
    def get_placement(self, ir_node_id: str) -> Optional[EntityPlacement]:
        """Get placement for an IR node."""
        return self.entity_placements.get(ir_node_id)
    
    def add_placement(self, placement: EntityPlacement) -> None:
        """Add an entity placement."""
        self.entity_placements[placement.ir_node_id] = placement
    
    def add_wire_connection(self, connection: WireConnection) -> None:
        """Add a wire connection."""
        self.wire_connections.append(connection)
    
    def add_power_pole(self, pole: PowerPolePlacement) -> None:
        """Add a power pole."""
        self.power_poles.append(pole)
```

### Step 2.2: Update layout module __init__.py ✅ Completed

```python
"""Layout planning module for the Factorio Circuit DSL."""

from .layout_engine import LayoutEngine
from .layout_plan import (
    LayoutPlan,
    EntityPlacement,
    WireConnection,
    PowerPolePlacement,
    SignalMaterialization,
)

__all__ = [
    "LayoutEngine",
    "LayoutPlan",
    "EntityPlacement",
    "WireConnection",
    "PowerPolePlacement",
    "SignalMaterialization",
]
```

### Step 2.3: Run tests ✅ Completed (2025-10-30, 116 passed)

```bash
python -m pytest tests/ -v
```

**Expected:** All tests pass. No behavior change (new data structures not used yet).

---

## Phase 3: Extract Signal Analysis

**Objective:** Move signal usage analysis and materialization logic to layout module.

### Step 3.1: Create signal_analyzer.py ✅ Completed

Create `dsl_compiler/src/layout/signal_analyzer.py` by extracting from `emission/signals.py`:

```python
"""Signal usage analysis and materialization decisions."""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from dsl_compiler.src.ast import SignalLiteral
from dsl_compiler.src.ir import (
    IRNode,
    IRValue,
    IR_Const,
    IR_Arith,
    IR_Decider,
    IR_MemCreate,
    IR_MemWrite,
    IR_PlaceEntity,
    IR_EntityPropWrite,
    IR_ConnectToWire,
    IR_WireMerge,
    SignalRef,
)
from dsl_compiler.src.semantic import DiagnosticCollector

# Copy SignalUsageEntry from emission/signals.py
@dataclass
class SignalUsageEntry:
    """Metadata describing how a logical signal is produced and consumed."""
    signal_id: str
    signal_type: Optional[str] = None
    producer: Optional[IRValue] = None
    consumers: Set[str] = field(default_factory=set)
    export_targets: Set[str] = field(default_factory=set)
    output_entities: Set[str] = field(default_factory=set)
    resolved_outputs: Dict[str, str] = field(default_factory=dict)
    source_ast: Optional[Any] = None
    literal_value: Optional[int] = None
    literal_declared_type: Optional[str] = None
    is_typed_literal: bool = False
    debug_label: Optional[str] = None
    debug_metadata: Dict[str, Any] = field(default_factory=dict)
    export_anchor_id: Optional[str] = None
    should_materialize: bool = True
    resolved_signal_name: Optional[str] = None
    resolved_signal_type: Optional[str] = None


class SignalAnalyzer:
    """Analyzes IR to determine signal usage patterns."""
    
    def __init__(self, diagnostics: DiagnosticCollector):
        self.diagnostics = diagnostics
        self.signal_usage: Dict[str, SignalUsageEntry] = {}
    
    def analyze(self, ir_operations: List[IRNode]) -> Dict[str, SignalUsageEntry]:
        """Analyze IR operations to build signal usage index."""
        self.signal_usage = {}
        
        # ... copy logic from BlueprintEmitter._analyze_signal_usage
        # This is the large method in emitter.py
        
        return self.signal_usage


class SignalMaterializer:
    """Decides when and how to materialize logical signals."""
    
    def __init__(
        self,
        signal_usage: Dict[str, SignalUsageEntry],
        signal_type_map: Dict[str, Any],
        diagnostics: DiagnosticCollector,
    ):
        self.signal_usage = signal_usage
        self.signal_type_map = signal_type_map
        self.diagnostics = diagnostics
    
    def finalize(self) -> None:
        """Populate materialization decisions."""
        # ... copy from emission/signals.py SignalMaterializer
        pass
    
    # ... copy all other methods from SignalMaterializer
```

**Action:** Copy the entire `SignalMaterializer` class and helper logic from `emission/signals.py`.

### Step 3.2: Update imports in signal_analyzer.py ✅ Completed

Make sure all imports work correctly. The file should have:
- IR imports from `dsl_compiler.src.ir`
- AST imports if needed
- Diagnostic imports from `dsl_compiler.src.semantic`

### Step 3.3: Update layout module __init__.py ✅ Completed

Add exports:
```python
from .signal_analyzer import SignalAnalyzer, SignalMaterializer, SignalUsageEntry
```

### Step 3.4: Update emitter.py to use layout module ✅ Completed

In `emission/emitter.py`:
```python
# OLD:
from .signals import SignalMaterializer, SignalUsageEntry

# NEW:
from dsl_compiler.src.layout import SignalAnalyzer, SignalMaterializer, SignalUsageEntry
```

### Step 3.5: Keep SignalGraph in emission temporarily ✅ Completed

**Note:** Keep `SignalGraph` and `EntityPlacement` in `emission/signals.py` for now. We'll move them in the next phase.

### Step 3.6: Run tests ✅ Completed (2025-10-30, 116 passed)

```bash
python -m pytest tests/ -v
```

**Expected:** All tests pass.

**Progress 2025-10-30 (night):** Full test suite green with the simplified emitter stack. Phase 7 complete.

---

## Phase 4: Extract Wire Planning and Connection Logic

**Objective:** Move wire color planning and connection building to layout module.

### Step 4.1: Create wire_router.py ✅ Completed

- Migrated the full wiring/coloring algorithms into `dsl_compiler/src/layout/wire_router.py`.
- Ensured layout package re-exports wire routing helpers for consumers.

Create `dsl_compiler/src/layout/wire_router.py` by moving content from `emission/wiring.py`:

```python
"""Wire routing and color assignment algorithms."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

# Copy all content from emission/wiring.py
# This includes:
# - WIRE_COLORS constant
# - CircuitEdge dataclass
# - ConflictNode, ConflictEdge, ColoringResult dataclasses
# - collect_circuit_edges function
# - detect_multi_source_conflicts function
# - plan_wire_colors function
```

### Step 4.2: Create connection_planner.py ✅ Completed

- Ported connection planning logic into `dsl_compiler/src/layout/connection_planner.py`, wiring in `LayoutPlan` and diagnostics integration.
- Added helpers to expose planned edges and colors for emission/tests.

Create `dsl_compiler/src/layout/connection_planner.py` by extracting from `emission/connection_builder.py`:

```python
"""Connection planning for wire routing."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from dsl_compiler.src.ir import SignalRef
from dsl_compiler.src.semantic import DiagnosticCollector

from .wire_router import CircuitEdge, plan_wire_colors, WIRE_COLORS
from .signal_analyzer import SignalUsageEntry
from .layout_plan import LayoutPlan, WireConnection


class ConnectionPlanner:
    """Plans all wire connections for a blueprint."""
    
    def __init__(
        self,
        layout_plan: LayoutPlan,
        signal_usage: Dict[str, SignalUsageEntry],
        diagnostics: DiagnosticCollector,
        max_wire_span: float = 9.0,
    ):
        self.layout_plan = layout_plan
        self.signal_usage = signal_usage
        self.diagnostics = diagnostics
        self.max_wire_span = max_wire_span
        
        self._circuit_edges: List[CircuitEdge] = []
        self._wire_color_assignments: Dict[Tuple[str, str], str] = {}
    
    def plan_connections(self) -> None:
        """Compute all wire connections with color assignments."""
        # Extract logic from ConnectionBuilder.prepare_wiring_plan
        # and create_circuit_connections
        pass
    
    def _compute_wire_distance(
        self,
        source_id: str,
        sink_id: str,
    ) -> float:
        """Compute distance between two placements."""
        source_placement = self.layout_plan.get_placement(source_id)
        sink_placement = self.layout_plan.get_placement(sink_id)
        
        if not source_placement or not sink_placement:
            return 0.0
        
        sx, sy = source_placement.position
        tx, ty = sink_placement.position
        return math.dist((sx, sy), (tx, ty))
```

**Action:** Extract and adapt connection planning logic from `connection_builder.py`.

### Step 4.3: Update layout module __init__.py ✅ Completed

- Exported the new planner and wire routing symbols so emission/tests consume the layout module exclusively.

```python
from .wire_router import WIRE_COLORS, CircuitEdge, plan_wire_colors
from .connection_planner import ConnectionPlanner
```

### Step 4.4: Update emission/emitter.py ✅ Completed

- Replaced `ConnectionBuilder` with the layout `ConnectionPlanner` and new in-emitter wiring helpers.
- Added placement registration syncing with `LayoutPlan`, relay insertion helpers, and connection realization pipeline.
- Updated power pole tracking to mirror into the layout plan.

Change imports and refactor to use ConnectionPlanner from layout module instead of ConnectionBuilder.

### Step 4.5: Delete emission/wiring.py and emission/connection_builder.py ✅ Completed

- Removed the legacy modules after all imports were redirected to the layout equivalents.
- Updated tests (`tests/test_wiring.py`, `tests/test_emit.py`) to target the new interfaces.

```bash
rm dsl_compiler/src/emission/wiring.py
rm dsl_compiler/src/emission/connection_builder.py
```

### Step 4.6: Run tests ✅ Completed (116 passed)

```bash
python -m pytest tests/ -v
```

**Expected:** All tests pass.

---

## Phase 5: Extract Power Pole Planning

**Objective:** Move power pole placement logic to layout module.

### Step 5.1: Create power_planner.py ✅ Completed

- Added `dsl_compiler/src/layout/power_planner.py` containing `PowerPlanner`, `PlannedPowerPole`, and the shared `POWER_POLE_CONFIG` constant.
- `PowerPlanner.plan_power_grid` now derives coverage tiles from the layout plan (using stored entity footprints), reserves slots via `LayoutEngine`, and guarantees connectivity by inserting bridge poles when necessary.
- Planner results are returned as `PlannedPowerPole` records; the emitter materializes physical entities from these plans.

### Step 5.2: Update layout module __init__.py ✅ Completed

- Exported `PowerPlanner`, `PlannedPowerPole`, and `POWER_POLE_CONFIG` so callers can coordinate with the layout power planning API.

### Step 5.3: Update emission/emitter.py ✅ Completed

- Removed in-emitter power pole planning helpers in favour of a new `_instantiate_power_pole` that consumes planned placements.
- `_record_entity_placement` now stores entity footprints in the mirrored layout plan to support planner coverage calculations.
- `_deploy_power_poles` instantiates `PowerPlanner`, requests a plan for the configured type, and materializes each placement while updating the layout plan and emitter bookkeeping.

### Step 5.4: Run tests ✅ Completed (116 passed)

```bash
python -m pytest tests/ -v
```

**Expected:** All tests pass.

---

## Phase 6: Create Layout Planner Orchestrator

**Objective:** Create main planner that orchestrates all layout planning.

### Step 6.1: Create planner.py ✅ Completed

- Added `dsl_compiler/src/layout/planner.py` defining `LayoutPlanner`, which orchestrates signal analysis, materialization, connection planner wiring, and optional power planning while resetting the layout engine for each run.
- Preserved the upcoming entity-placement extraction by leaving `_place_entities` as an explicit `NotImplementedError`, making the remaining migration work obvious for the next phase.

### Step 6.2: Update layout module __init__.py ✅ Completed

- Re-exported `LayoutPlanner` alongside existing layout utilities so callers can construct the orchestrator from the layout package root.

### Step 6.3: Run tests ✅ Completed (116 passed)

```bash
python -m pytest tests/ -v
```

**Expected:** Planner scaffolding coexists with the legacy emitter (planner not yet invoked).

---

## Phase 7: Refactor Emission to Consume LayoutPlan

**Objective:** Make emission a thin wrapper that materializes the LayoutPlan.

### Step 7.1: Create new simplified emitter

Rewrite `emission/emitter.py` to consume LayoutPlan:

```python
"""Simplified blueprint emission using LayoutPlan."""

from typing import Dict, List, Tuple

from draftsman.blueprintable import Blueprint
from draftsman.entity import new_entity

from dsl_compiler.src.ir import IRNode
from dsl_compiler.src.semantic import DiagnosticCollector
from dsl_compiler.src.layout import LayoutPlan, LayoutPlanner


class BlueprintEmitter:
    """Materializes LayoutPlan into Factorio blueprint."""
    
    def __init__(self, signal_type_map: Dict[str, str]):
        self.signal_type_map = signal_type_map
        self.diagnostics = DiagnosticCollector()
        self.blueprint = Blueprint()
    
    def emit_from_plan(self, layout_plan: LayoutPlan) -> Blueprint:
        """Emit blueprint from a complete layout plan."""
        self.blueprint = Blueprint()
        self.blueprint.label = layout_plan.blueprint_label
        self.blueprint.description = layout_plan.blueprint_description
        self.blueprint.version = (2, 0)
        
        # Phase 1: Create all entities
        self._materialize_entities(layout_plan)
        
        # Phase 2: Create all wire connections
        self._materialize_connections(layout_plan)
        
        # Phase 3: Create power grid
        self._materialize_power_grid(layout_plan)
        
        return self.blueprint
    
    def _materialize_entities(self, layout_plan: LayoutPlan) -> None:
        """Create draftsman entities from placements."""
        for placement in layout_plan.entity_placements.values():
            entity = new_entity(placement.entity_type)
            entity.tile_position = placement.position
            
            # Apply properties
            for prop_name, prop_value in placement.properties.items():
                if hasattr(entity, prop_name):
                    setattr(entity, prop_name, prop_value)
            
            self.blueprint.entities.append(entity, copy=False)
    
    def _materialize_connections(self, layout_plan: LayoutPlan) -> None:
        """Create wire connections."""
        for connection in layout_plan.wire_connections:
            # Look up entities by ID
            source_placement = layout_plan.get_placement(connection.source_entity_id)
            sink_placement = layout_plan.get_placement(connection.sink_entity_id)
            
            if not source_placement or not sink_placement:
                continue
            
            # Add connection
            kwargs = {
                "color": connection.wire_color,
                # ... entity lookup
            }
            self.blueprint.add_circuit_connection(**kwargs)
    
    def _materialize_power_grid(self, layout_plan: LayoutPlan) -> None:
        """Create power poles and connections."""
        # Create power pole entities
        for pole in layout_plan.power_poles:
            entity = new_entity(pole.pole_type)
            entity.tile_position = pole.position
            self.blueprint.entities.append(entity, copy=False)
        
        # Generate power connections
        if layout_plan.power_poles:
            self.blueprint.generate_power_connections()


def emit_blueprint_string(
    ir_operations: List[IRNode],
    label: str,
    signal_type_map: Dict[str, str],
    power_pole_type: Optional[str] = None,
) -> Tuple[str, DiagnosticCollector]:
    """
    High-level API: Plan layout and emit blueprint string.
    """
    # Phase 1: Layout planning
    planner = LayoutPlanner(
        signal_type_map,
        power_pole_type=power_pole_type,
    )
    layout_plan = planner.plan_layout(ir_operations, label)
    
    # Phase 2: Emission
    emitter = BlueprintEmitter(signal_type_map)
    blueprint = emitter.emit_from_plan(layout_plan)
    
    # Phase 3: Serialize
    blueprint_string = blueprint.to_string()
    
    # Combine diagnostics
    all_diagnostics = DiagnosticCollector()
    all_diagnostics.diagnostics.extend(planner.diagnostics.diagnostics)
    all_diagnostics.diagnostics.extend(emitter.diagnostics.diagnostics)
    
    return blueprint_string, all_diagnostics
```

**Progress 2025-10-30:** Created layout-side support modules (`signal_graph.py`, `signal_resolver.py`, `memory_builder.py`) to decouple planning logic from the legacy emitter. Full emitter rewrite still pending.

**Progress 2025-10-30 (continued):** Introduced transitional `LegacyBlueprintEmitter` for planning and added a new thin `BlueprintEmitter.emit_from_plan` pipeline that consumes `LayoutPlan`. High-level helpers (`emit_blueprint`, `emit_blueprint_string`) now run layout planning first.

**Progress 2025-10-30 (late evening):** Step 7.2 complete — `emission/entity_emitter.py` now materializes entities directly from `LayoutPlan` via `PlanEntityEmitter`, and the legacy planning helpers were moved under `layout/`. Step 7.3 also complete with the removal of the old `signals.py` and `signal_resolver.py` from the emission package. All tests pass.

**Progress 2025-10-31 early AM:** Phase 8 complete — `compile.py` now orchestrates layout planning via `LayoutPlanner` before materializing with `BlueprintEmitter`, the emission package exports are reduced to the new public surface, tests updated to target the new API, and CLI compilation of `tests/sample_programs/01_basic_arithmetic.fcdsl` succeeds. Full test suite remains green.

**Progress 2025-10-30 (late evening):** Step 7.2 complete — `emission/entity_emitter.py` now materializes entities directly from `LayoutPlan` via `PlanEntityEmitter`, and the legacy planning helpers were moved under `layout/`. Step 7.3 also complete with the removal of the old `signals.py` and `signal_resolver.py` from the emission package.

### Step 7.2: Simplify entity_emitter.py

Refactor `emission/entity_emitter.py` to work with LayoutPlan. It should only handle entity creation, not placement.

### Step 7.3: Remove obsolete emission files

After confirming new emitter works:
```bash
# Keep only:
# - emitter.py (rewritten)
# - entity_emitter.py (simplified)
# - memory.py (simplified)
# - debug_format.py (utility, can stay)

# Remove:
rm dsl_compiler/src/emission/signals.py
rm dsl_compiler/src/emission/signal_resolver.py
```

### Step 7.4: Run tests

```bash
python -m pytest tests/ -v
```

**Expected:** All tests pass.

---

## Phase 8: Integrate Layout Step into Compilation Pipeline

**Objective:** Add layout planning as explicit step in compile.py.

### Step 8.1: Update compile.py

Modify `compile.py` to add layout step:

```python
def compile_dsl_file(
    input_path: Path,
    strict_types: bool = False,
    program_name: str = None,
    optimize: bool = True,
    explain: bool = False,
    power_pole_type: str | None = None,
) -> tuple[bool, str, list]:
    """Compile DSL file to blueprint string."""
    
    # ... existing parsing and semantic analysis ...
    
    # IR generation
    ir_operations, lowering_diagnostics, signal_type_map = lower_program(
        program, analyzer
    )
    
    if lowering_diagnostics.has_errors():
        all_diagnostics.extend(lowering_diagnostics.get_messages())
        return False, "IR lowering failed", all_diagnostics
    
    all_diagnostics.extend(lowering_diagnostics.get_messages())
    
    # Optimize
    if optimize:
        ir_operations = CSEOptimizer().optimize(ir_operations)
    
    # **NEW: Layout Planning**
    from dsl_compiler.src.layout import LayoutPlanner
    
    planner = LayoutPlanner(
        signal_type_map,
        power_pole_type=power_pole_type,
    )
    
    layout_plan = planner.plan_layout(
        ir_operations,
        f"{program_name} Blueprint"
    )
    
    if planner.diagnostics.has_errors():
        all_diagnostics.extend(planner.diagnostics.get_messages())
        return False, "Layout planning failed", all_diagnostics
    
    all_diagnostics.extend(planner.diagnostics.get_messages())
    
    # **NEW: Blueprint Emission (now just materialization)**
    from dsl_compiler.src.emission import BlueprintEmitter
    
    emitter = BlueprintEmitter(signal_type_map)
    blueprint = emitter.emit_from_plan(layout_plan)
    
    if emitter.diagnostics.has_errors():
        all_diagnostics.extend(emitter.diagnostics.get_messages())
        return False, "Blueprint emission failed", all_diagnostics
    
    all_diagnostics.extend(emitter.diagnostics.get_messages())
    
    # Serialize
    blueprint_string = blueprint.to_string()
    
    # Validate
    if not blueprint_string or len(blueprint_string) < 10:
        return False, "Invalid blueprint string generated", all_diagnostics
    
    return True, blueprint_string, all_diagnostics
```

### Step 8.2: Update emission __init__.py exports

```python
from .emitter import BlueprintEmitter, emit_blueprint_string

__all__ = [
    "BlueprintEmitter",
    "emit_blueprint_string",
]
```

### Step 8.3: Run full test suite

```bash
python -m pytest tests/ -v
```

### Step 8.4: Test compilation end-to-end

```bash
python compile.py tests/sample_programs/simple.fcdsl
```

**Expected:** Blueprint generated successfully.

---

## Phase 9: Verification and Cleanup

### Step 9.1: Verify file structure ✅ Completed (2025-10-31)

**Layout module should contain:**
- `__init__.py`
- `planner.py` - main orchestrator
- `layout_engine.py` - spatial placement
- `layout_plan.py` - data structures
- `signal_analyzer.py` - signal usage analysis
- `connection_planner.py` - wire planning
- `wire_router.py` - wire coloring
- `power_planner.py` - power pole planning

**Emission module should contain:**
- `__init__.py`
- `emitter.py` - simplified blueprint materializer
- `entity_emitter.py` - entity creation helpers (optional)
- `memory.py` - memory circuit helpers (optional)
- `debug_format.py` - debug utilities (optional)

**Status:** Verified. Removed unused `layout.debug`, `layout.entity_emitter`, and
`layout.memory_builder` modules; emission `debug_format` now hosts the debug
annotation helper.

### Step 9.2: Remove unused imports ✅ Completed (2025-10-31)

Search for and remove any unused imports in both modules.

**Status:** Cleaned imports after module removals and re-routed debug helper
imports to `dsl_compiler.src.emission.debug_format`.

### Step 9.3: Update documentation ✅ Completed (2025-10-31)

Add docstrings explaining the new architecture:

```python
# At top of dsl_compiler/src/layout/__init__.py
"""
Layout Planning Module
=====================

This module handles all physical layout planning for circuit blueprints:

1. Signal Analysis - determines which signals need materialization
2. Entity Placement - assigns x,y coordinates to all entities
3. Connection Planning - determines wire routing and colors
4. Power Planning - places power poles to cover entities

The output is a LayoutPlan which the emission module materializes
into a Factorio blueprint.
"""
```

### Step 9.4: Run complete test suite ✅ Completed (2025-10-31)

```bash
python -m pytest tests/ -v --cov=dsl_compiler
```

**Status:** 116 passing tests; coverage run recorded. Existing Draftsman
warnings retained.

### Step 9.5: Test sample programs ✅ Completed (2025-10-31)

```bash
for f in tests/sample_programs/*.fcdsl; do
    echo "Testing $f"
    python compile.py "$f" || exit 1
done
```

**Status:** All sample programs compile successfully via `compile.py` (existing
Draftsman warnings only).

---

## Critical Success Criteria

After each phase, verify:

1. ✅ **All tests pass** - `pytest tests/ -v`
2. ✅ **No import errors** - Python can import all modules
3. ✅ **Sample programs compile** - Can generate blueprints
4. ✅ **No runtime errors** - Programs execute without exceptions
5. ✅ **Diagnostic messages preserved** - Error reporting still works
6. ✅ **Blueprint functionality** - Generated blueprints are valid

## Rollback Strategy

If any phase fails:

1. Revert changes: `git checkout -- .`
2. Review the specific step that failed
3. Address the issue
4. Re-run from that step

## Key Principles

1. **Never break the main branch** - Each step must be testable
2. **Incremental changes** - Small, verifiable steps
3. **Preserve behavior** - Output blueprints must be identical
4. **Keep tests passing** - Run tests after every step
5. **Document as you go** - Update comments and docstrings

## Testing Strategy

For each phase:
```bash
# Run basic tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_emission.py -v

# Test sample program
python compile.py tests/sample_programs/simple.fcdsl -o /tmp/test.blueprint

# Verify blueprint validity (length check)
test $(cat /tmp/test.blueprint | wc -c) -gt 100
```

---

This guide provides a methodical approach that maintains working state at each step. The code-generating model should follow each step precisely, running tests between steps, and not proceed if tests fail.