# Factorio DSL Compiler: Layout Pipeline Refactoring Plan

## Executive Summary

**Goal**: Extract all layout and positioning logic from the `BlueprintEmitter` into a dedicated **Layout Pass** that runs between IR generation and emission. The emitter becomes a lightweight translator that performs 1:1 IR-to-Blueprint entity mapping.

**Current Problem**: `BlueprintEmitter` has ~1500 lines doing layout, power poles, memory building, signal resolution, entity creation, and connection planning—violating separation of concerns.

**Solution**: Insert a new pipeline stage that augments IR nodes with position metadata, handles clustering, edge zones, and infrastructure planning before emission.

---

## Progress Tracker

- Phase 1 (IR metadata & node scaffolding): **Complete** – `ir/nodes.py` now carries layout fields and infrastructure node definitions.
- Phase 2 (Layout package scaffolding): **Complete** – `dsl_compiler/src/layout/` houses the new pass façade and helpers.
- Phase 3 (Clustering analyzer): **Complete** – memory, merge, and entity clusters implemented with tests in `tests/test_layout_pass.py`.
- Phase 4 (Layout pass core logic): **Complete** – `layout_pass.py` orchestrates zoning, positioning, and export anchors.
- Phase 5 (Infrastructure planner): **Complete** – relay and power planning moved to `layout/infrastructure.py`.
- Phase 6 (Emitter refactor): **Complete** – `BlueprintEmitter` consumes layout metadata and remaps relay paths.
- Phase 7 (Entity emitter refactor): **Complete** – entity placement uses layout positions and registers aliases.
- Phase 8 (Pipeline integration): **Complete** – `compile.py` and emission entry points run the layout pass prior to emission.
- Phase 9 (Test coverage updates): **In Progress** – baseline layout tests exist; end-to-end regressions still failing on complex samples.
- Phase 10 (Documentation refresh): **Pending** – specs still describe legacy emitter-driven layout.

## Phase 1: Foundation & Data Structure Changes

### 1.1 Augment IR Nodes with Position Metadata

**Status: Complete**

**File**: `dsl_compiler/src/ir/nodes.py`

**Changes**:
```python
class IRNode(ABC):
    def __init__(self, node_id: str, source_ast: Optional[ASTNode] = None):
        self.node_id = node_id
        self.source_ast = source_ast
        self.debug_metadata: Dict[str, Any] = {}
        # NEW: Layout metadata
        self.layout_position: Optional[Tuple[int, int]] = None
        self.layout_zone: Optional[str] = None  # 'north_literals', 'south_exports', 'main', etc.
        self.layout_cluster: Optional[str] = None  # For grouping related nodes
```

**Rationale**: Position information belongs with the IR node, not computed on-the-fly during emission.

---

### 1.2 Create Layout Pass Infrastructure

**Status: Complete**

**New File**: `dsl_compiler/src/layout/__init__.py`

```python
"""Layout pipeline for positioning IR operations before emission."""

from .layout_pass import LayoutPass, layout_ir_operations
from .clustering import ClusterAnalyzer
from .infrastructure import InfrastructurePlanner

__all__ = [
    "LayoutPass",
    "layout_ir_operations",
    "ClusterAnalyzer", 
    "InfrastructurePlanner",
]
```

---

### 1.3 Create Core LayoutPass Class

**Status: Complete**

**New File**: `dsl_compiler/src/layout/layout_pass.py`

**Structure**:
```python
class LayoutPass:
    """Main orchestrator for IR positioning pipeline."""
    
    def __init__(self, signal_usage: Dict[str, SignalUsageEntry], signal_type_map: Dict[str, str]):
        self.signal_usage = signal_usage
        self.signal_type_map = signal_type_map
        self.diagnostics = DiagnosticCollector()
        
        # Sub-components
        self.engine = LayoutEngine()  # Reuse existing
        self.cluster_analyzer = ClusterAnalyzer()
        self.infra_planner = InfrastructurePlanner()
        
        # Tracking
        self.positioned_nodes: Dict[str, IRNode] = {}
        self.memory_clusters: Dict[str, List[str]] = {}
        self.literal_zone_nodes: List[str] = []
        self.export_zone_nodes: List[str] = []
    
    def position_ir_operations(self, ir_operations: List[IRNode]) -> List[IRNode]:
        """Main entry point: augment IR with positions."""
        self._analyze_clustering(ir_operations)
        self._assign_zones(ir_operations)
        self._position_literals()
        self._position_memory_clusters()
        self._position_main_operations(ir_operations)
        self._position_exports()
        self._plan_infrastructure()
        return ir_operations
```

**Key Methods**:
- `_analyze_clustering()`: Group related operations (memory modules, wire merges)
- `_assign_zones()`: Classify nodes into north/south/main zones
- `_position_literals()`: Place constants at north edge adaptively
- `_position_memory_clusters()`: Cluster SR latches together
- `_position_main_operations()`: Position computations with locality awareness
- `_position_exports()`: Place export anchors at south edge adaptively
- `_plan_infrastructure()`: Insert power poles and wire relays

---

## Phase 2: Implement Layout Sub-Components

### 2.1 Clustering Analyzer

**Status: Complete**

**New File**: `dsl_compiler/src/layout/clustering.py`

```python
class ClusterAnalyzer:
    """Identifies groups of related IR operations that should be spatially clustered."""
    
    def analyze(self, ir_operations: List[IRNode]) -> Dict[str, List[str]]:
        """Returns cluster_id -> [node_ids] mapping."""
        clusters = {}
        
        # Cluster 1: Memory modules (write_gate + hold_gate for each memory)
        memory_clusters = self._find_memory_clusters(ir_operations)
        clusters.update(memory_clusters)
        
        # Cluster 2: Wire merge junctions (all sources feeding a merge)
        merge_clusters = self._find_merge_clusters(ir_operations)
        clusters.update(merge_clusters)
        
        # Cluster 3: Entity property readers/writers for same entity
        entity_clusters = self._find_entity_clusters(ir_operations)
        clusters.update(entity_clusters)
        
        return clusters
```

**Rationale**: Clustering keeps related operations close, reducing wire distances and improving blueprint readability.

---

### 2.2 Infrastructure Planner

**Status: Complete**

**New File**: `dsl_compiler/src/layout/infrastructure.py`

```python
class InfrastructurePlanner:
    """Plans power pole and wire relay insertion points."""
    
    def __init__(self, wire_relay_options: WireRelayOptions):
        self.relay_options = wire_relay_options
        self.power_pole_config = None  # Set if power poles enabled
        
    def plan_infrastructure(
        self,
        positioned_nodes: Dict[str, IRNode],
        signal_graph: SignalGraph,
    ) -> List[IRNode]:
        """Returns new IR nodes for power poles and wire relays."""
        infrastructure_nodes = []
        
        # Plan wire relays
        if self.relay_options.enabled:
            relay_nodes = self._plan_wire_relays(positioned_nodes, signal_graph)
            infrastructure_nodes.extend(relay_nodes)
        
        # Plan power poles
        if self.power_pole_config:
            pole_nodes = self._plan_power_poles(positioned_nodes)
            infrastructure_nodes.extend(pole_nodes)
        
        return infrastructure_nodes
    
    def _plan_wire_relays(self, ...) -> List[IRNode]:
        """Insert relay poles where signal paths exceed wire reach."""
        # Analyze positioned nodes, find long-distance connections
        # Create IR_Infrastructure nodes for medium-electric-pole placement
        pass
    
    def _plan_power_poles(self, ...) -> List[IRNode]:
        """Insert power poles to cover all positioned entities."""
        # Use existing grid coverage algorithm
        # Return IR_Infrastructure nodes for power pole placement
        pass
```

**New IR Node**:
```python
class IR_Infrastructure(IREffect):
    """Infrastructure entity (power pole, wire relay)."""
    def __init__(self, entity_type: str, x: int, y: int, role: str):
        super().__init__(f"infra_{entity_type}_{x}_{y}")
        self.entity_type = entity_type  # "medium-electric-pole", etc.
        self.x = x
        self.y = y
        self.role = role  # "power", "wire_relay"
```

---

### 2.3 Adaptive Edge Zone Placement

**Enhancement to LayoutPass**:

```python
def _position_literals(self):
    """Place literal constants along north edge, adapting row count to width."""
    literals = [op for op in self.ir_ops if isinstance(op, IR_Const) and self._should_materialize(op)]
    
    if not literals:
        return
    
    # Adaptive row allocation: if >20 literals, use multiple rows
    row_capacity = 20
    num_rows = max(1, math.ceil(len(literals) / row_capacity))
    
    base_y = -self.engine.row_height * num_rows
    
    for idx, lit_op in enumerate(literals):
        row = idx // row_capacity
        col = idx % row_capacity
        x = col * self.engine.entity_spacing
        y = base_y + row * self.engine.row_height
        
        lit_op.layout_position = (x, y)
        lit_op.layout_zone = "north_literals"
        self.literal_zone_nodes.append(lit_op.node_id)

def _position_exports(self):
    """Place export anchors along south edge, similar adaptive logic."""
    # Same pattern as literals but at positive Y
    pass
```

**Rationale**: Currently hardcoded single row; adaptive multi-row handles wide blueprints gracefully.

---

## Phase 3: Memory Clustering Implementation

### 3.1 Memory Cluster Detection

**Status: Complete**

**In `ClusterAnalyzer`**:

```python
def _find_memory_clusters(self, ir_operations: List[IRNode]) -> Dict[str, List[str]]:
    """Group memory create/read/write operations by memory_id."""
    clusters = {}
    
    for op in ir_operations:
        if isinstance(op, (IR_MemCreate, IR_MemRead, IR_MemWrite)):
            memory_id = op.memory_id
            cluster_key = f"memory_{memory_id}"
            
            if cluster_key not in clusters:
                clusters[cluster_key] = []
            
            clusters[cluster_key].append(op.node_id)
    
    return clusters
```

---

### 3.2 Memory Cluster Positioning

**Status: Complete**

**In `LayoutPass`**:

```python
def _position_memory_clusters(self):
    """Position memory modules as tight clusters."""
    for cluster_id, node_ids in self.memory_clusters.items():
        # Find memory module components
        create_op = next(op for op in self.ir_ops if op.node_id in node_ids and isinstance(op, IR_MemCreate))
        
        # Allocate cluster anchor point
        anchor = self.engine.get_next_position(footprint=(4, 3))  # Reserve 4x3 for SR latch
        
        # Position write gate and hold gate relative to anchor
        # (This assumes memory builder creates these during emission prep - see Phase 4)
        write_gate_pos = anchor
        hold_gate_pos = (anchor[0] + 2, anchor[1])
        
        # Annotate IR nodes with positions
        create_op.layout_position = anchor
        create_op.layout_cluster = cluster_id
        
        # Position read/write operations near the cluster
        for node_id in node_ids:
            op = self.positioned_nodes[node_id]
            if isinstance(op, (IR_MemRead, IR_MemWrite)):
                op.layout_position = self.engine.reserve_near(anchor, max_radius=6)
                op.layout_cluster = cluster_id
```

**Rationale**: SR latches stay together, reads/writes position nearby for short wires.

---

## Phase 4: Refactor Emitter to Use Positions

### 4.1 Simplify BlueprintEmitter

**Status: Complete**

**File**: `dsl_compiler/src/emission/emitter.py`

**Major Changes**:

1. **Remove** `self.layout = LayoutEngine()` - positions come from IR
2. **Remove** `_place_entity()`, `_place_entity_in_zone()`, `_allocate_position()` - use `op.layout_position`
3. **Remove** `_deploy_power_poles()`, `_connect_power_grid()` - handled by layout pass
4. **Simplify** `_ensure_export_anchors()` - just emit, don't position

**New Entry Point**:

```python
def emit_blueprint(self, ir_operations: List[IRNode]) -> Blueprint:
    """Convert positioned IR to blueprint. Expects IR with layout_position set."""
    
    # Validate all operations have positions
    if not self._validate_positions(ir_operations):
        self.diagnostics.error("IR operations missing layout positions")
        return self.blueprint
    
    # Simple emission loop
    for op in ir_operations:
        self.emit_ir_operation(op)
    
    # Wire connections (positions already known)
    self.connection_builder.create_circuit_connections()
    
    return self.blueprint

def _validate_positions(self, ir_operations: List[IRNode]) -> bool:
    """Ensure all IR nodes have layout_position set."""
    for op in ir_operations:
        if isinstance(op, (IR_Const, IR_Arith, IR_Decider, IR_MemCreate, IR_PlaceEntity, IR_Infrastructure)):
            if op.layout_position is None:
                self.diagnostics.error(f"Missing layout position for {op.node_id}")
                return False
    return True
```

---

### 4.2 Simplify Entity Emitter

**Status: Complete**

**File**: `dsl_compiler/src/emission/entity_emitter.py`

**Changes**:

```python
def emit_constant(self, op: IR_Const):
    """Emit constant combinator using pre-assigned position."""
    combinator = new_entity("constant-combinator")
    
    # Use position from layout pass
    assert op.layout_position is not None, f"Constant {op.node_id} missing position"
    combinator.tile_position = op.layout_position
    
    # Configure signal value (unchanged)
    section = combinator.add_section()
    # ... existing signal setup ...
    
    combinator = self._add_entity(combinator)
    # ... rest unchanged ...
```

**Apply Same Pattern** to:
- `emit_arithmetic()`
- `emit_decider()`
- `emit_memory_create()`
- `emit_place_entity()` - validate user position vs layout position

---

### 4.3 Handle Infrastructure Nodes

**Status: Complete**

**In `entity_emitter.py`**:

```python
def emit_infrastructure(self, op: IR_Infrastructure):
    """Emit power pole or wire relay."""
    entity = new_entity(op.entity_type)
    entity.tile_position = (op.x, op.y)
    entity = self._add_entity(entity)
    
    placement = EntityPlacement(
        entity=entity,
        entity_id=op.node_id,
        position=(op.x, op.y),
        output_signals={},
        input_signals={},
        role=op.role,
    )
    self.entities[op.node_id] = placement
```

---

## Phase 5: Update Compilation Pipeline

### 5.1 Modify compile.py

**Status: Complete**

**File**: `compile.py`

**Insert Layout Pass**:

```python
def compile_dsl_file(...):
    # ... existing parsing, semantic analysis, lowering ...
    
    # IR generation
    ir_operations, lowering_diagnostics, signal_type_map = lower_program(program, analyzer)
    
    # === NEW: Layout Pass ===
    from dsl_compiler.src.layout import layout_ir_operations
    
    ir_operations, layout_diagnostics = layout_ir_operations(
        ir_operations,
        signal_type_map=signal_type_map,
        signal_usage=analyzer.signal_usage,  # Pass from semantic analysis
        power_pole_type=power_pole_type,
        wire_relay_options=WireRelayOptions(...),
    )
    
    all_diagnostics.extend(layout_diagnostics.get_messages())
    if layout_diagnostics.has_errors():
        return False, "Layout planning failed", all_diagnostics
    
    # === Modified: Emission (now lightweight) ===
    blueprint_string, emit_diagnostics = emit_blueprint_string(
        ir_operations,  # Now has positions
        f"{program_name} Blueprint",
        signal_type_map,
        # power_pole_type removed - handled in layout
    )
```

---

### 5.2 Create Layout Pass Entry Point

**Status: Complete**

**File**: `dsl_compiler/src/layout/layout_pass.py`

```python
def layout_ir_operations(
    ir_operations: List[IRNode],
    signal_type_map: Dict[str, str],
    signal_usage: Dict[str, SignalUsageEntry],
    power_pole_type: Optional[str] = None,
    wire_relay_options: Optional[WireRelayOptions] = None,
) -> Tuple[List[IRNode], DiagnosticCollector]:
    """Public API: position all IR operations and insert infrastructure."""
    
    layout_pass = LayoutPass(signal_usage, signal_type_map)
    
    # Configure infrastructure
    if power_pole_type:
        layout_pass.infra_planner.power_pole_config = POWER_POLE_CONFIG[power_pole_type]
    
    if wire_relay_options:
        layout_pass.infra_planner.relay_options = wire_relay_options
    
    # Run layout
    positioned_ir = layout_pass.position_ir_operations(ir_operations)
    
    return positioned_ir, layout_pass.diagnostics
```

---

## Phase 6: Testing & Validation

### 6.1 Update Test Expectations

**Status: In Progress** – initial layout-focused tests exist, but end-to-end fixtures still require updates once layout stability is restored.

**Files**: `tests/test_*.py`

**Changes**:
- Tests expecting specific entity counts may break (power poles now planned in layout)
- Tests validating positions need updating (edge zones now adaptive)
- Add new tests for layout pass in isolation

**New Test File**: `tests/test_layout_pass.py`

```python
def test_layout_pass_positions_all_nodes():
    """Verify every IR node gets a position."""
    ir_ops = [...]
    positioned, diags = layout_ir_operations(ir_ops, ...)
    for op in positioned:
        if isinstance(op, (IR_Const, IR_Arith, ...)):
            assert op.layout_position is not None

def test_memory_clustering():
    """Verify memory operations cluster together."""
    ir_ops = create_memory_test_ir()
    positioned, diags = layout_ir_operations(ir_ops, ...)
    
    # Find memory cluster positions
    create_pos = find_op(positioned, "IR_MemCreate").layout_position
    read_pos = find_op(positioned, "IR_MemRead").layout_position
    write_pos = find_op(positioned, "IR_MemWrite").layout_position
    
    # Verify proximity
    assert distance(create_pos, read_pos) < 10
    assert distance(create_pos, write_pos) < 10
```

---

## Phase 7: Documentation Updates

### 7.1 Update COMPILER_SPEC.md

**Status: Pending**

**Section**: End-to-End Control Flow

**New Step 6.5**:
```markdown
6. **Layout Planning (`layout/`)** – Assigns (x, y) coordinates to all IR operations and plans infrastructure placement.
   - `layout_pass.py` orchestrates positioning
   - `clustering.py` groups related operations  
   - `infrastructure.py` inserts power poles and wire relays
   - Augments IR nodes with `layout_position`, `layout_zone`, `layout_cluster` metadata
```

**Section**: Module Reference

**Add**:
```markdown
### Layout Package (`dsl_compiler/src/layout`)
- `layout_pass.py` – Main orchestrator for positioning IR operations
- `clustering.py` – Identifies groups of related operations for spatial clustering
- `infrastructure.py` – Plans power pole and wire relay insertion points
```

---

### 7.2 Update LANGUAGE_SPEC.md

**Status: Pending**

**Section**: Compilation Pipeline

**Update**:
```markdown
1. **Lexical Analysis**: Source → Tokens
2. **Parsing**: Tokens → AST
3. **Semantic Analysis**: Type inference, symbol resolution, validation
4. **IR Generation**: AST → Intermediate Representation
5. **Layout Planning**: IR → Positioned IR (with coordinates and infrastructure)
6. **Blueprint Emission**: Positioned IR → Factorio blueprint JSON
```

---

## Execution Order for Code Generation Agent (Historical Reference)

### Step 1: IR Node Augmentation (30 min)
**Status: Complete**
- Modify `ir/nodes.py` to add `layout_position`, `layout_zone`, `layout_cluster` fields
- Add `IR_Infrastructure` node class

### Step 2: Create Layout Package Structure (15 min)
**Status: Complete**
- Create `dsl_compiler/src/layout/` directory
- Create `__init__.py` with exports
- Create empty `layout_pass.py`, `clustering.py`, `infrastructure.py`

### Step 3: Implement ClusterAnalyzer (45 min)
**Status: Complete**
- Implement `clustering.py` with memory, merge, entity cluster detection
- Add tests in `tests/test_layout_clustering.py`

### Step 4: Implement LayoutPass Core (2 hours)
**Status: Complete**
- Implement `layout_pass.py` with zone assignment and positioning logic
- Reuse `LayoutEngine` from emission for low-level grid management
- Implement adaptive multi-row edge placement

### Step 5: Implement InfrastructurePlanner (1.5 hours)
**Status: Complete**
- Move power pole logic from `emitter.py` to `infrastructure.py`
- Move wire relay logic from `connection_builder.py` to `infrastructure.py`
- Return `IR_Infrastructure` nodes instead of direct entity creation

### Step 6: Refactor BlueprintEmitter (2 hours)
**Status: Complete**
- Remove layout, power pole, positioning methods
- Update `emit_blueprint()` to expect positioned IR
- Add position validation
- Remove `_place_entity()`, `_place_entity_in_zone()`, etc.

### Step 7: Refactor EntityEmitter (1 hour)
**Status: Complete**
- Update all `emit_*` methods to use `op.layout_position`
- Add `emit_infrastructure()` method
- Remove position computation logic

### Step 8: Update Compilation Pipeline (30 min)
**Status: Complete**
- Modify `compile.py` to insert layout pass
- Update `emit_blueprint_string()` signature (remove power_pole_type)

### Step 9: Fix Tests (1 hour)
**Status: In Progress**
- Update test expectations for entity counts
- Add new layout pass tests
- Verify end-to-end compilation still works

### Step 10: Documentation (30 min)
**Status: Pending**
- Update `COMPILER_SPEC.md` with layout pipeline step
- Update `LANGUAGE_SPEC.md` compilation model section

---

## Expected Outcomes

### Benefits
1. **Separation of Concerns**: Layout logic isolated from emission
2. **Testability**: Layout pass testable independently
3. **Flexibility**: Easy to add new layout strategies (e.g., minimum wire length)
4. **Maintainability**: Emitter becomes <500 lines of straightforward translation
5. **Extensibility**: New clustering strategies easy to add

### File Size Changes
- `emitter.py`: 1500 → ~400 lines (70% reduction)
- New `layout_pass.py`: ~600 lines
- New `clustering.py`: ~300 lines
- New `infrastructure.py`: ~400 lines

### Performance Impact
- Negligible: One extra pass over IR (linear time)
- Potential improvement: Better spatial locality reduces wire distance calculations

---

## Risk Mitigation

### Risk 1: Breaking Existing Tests
**Mitigation**: Run tests after each step; fix incrementally

### Risk 2: Memory Builder Complexity
**Mitigation**: Keep `MemoryCircuitBuilder` in emission package initially; refactor later if needed

### Risk 3: Position Conflicts
**Mitigation**: Layout pass validates no overlaps before returning

### Risk 4: Wire Relay Coordination
**Mitigation**: `ConnectionBuilder` reads positions from IR nodes; no logic change needed

---

## Phase 8: Post-Refactor Reliability Fixes (Active)

### 8.1 Tighten Relay Segmentation Heuristics
- **Objective**: Ensure `InfrastructurePlanner` breaks long edges (e.g., `arith_116` → `arith_15`) into sub-span segments below the 9-tile reach.
- **Plan**: Enhance `_plan_wire_relays` to consider intermediate anchors based on actual routed distance, accounting for obstacles and reserving candidate tiles earlier to avoid placement failure cascades.

### 8.2 Expand Layout Spacing Around Mixed Clusters
- **Objective**: Adjust layout spacing between combinator clusters, assemblers, and infrastructure to prevent Draftsman overlap exceptions.
- **Plan**: Update `LayoutPass` footprint estimation and positioning to expand padding for mixed prototypes, and reserve exclusion zones when memory clusters neighbor large entities.

### 8.3 Reduce Wire Color Conflicts in Memory Write Nets
- **Objective**: Lower multi-producer pressure on memory write channels so the two-color planner maintains bipartite routing instead of downgrading to single-channel wiring.
- **Plan**: Revisit merge layout and signal graph hints to distribute producers, introduce relay-mediated fan-in points, or split write orchestration across dedicated combinators before wiring.




This refactoring plan provides a clear, executable path to separate layout from emission while improving code quality and maintainability. Each phase builds on the previous one with minimal disruption to existing functionality.