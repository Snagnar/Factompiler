# Layout Module

## Purpose
Converts IR operations into physical layouts - determines where to place combinators, how to wire them, and how to power them. This is the most complex stage of compilation.

## Files

### planner.py
**Main Layout Orchestrator**
- `LayoutPlanner`: Coordinates physical layout planning
  - Entry: `plan_layout(ir_ops)` → `LayoutPlan`
  - Delegates to specialized planners
  - Manages entity placement, wiring, and power

### layout_plan.py
**Layout Data Structure**
- `LayoutPlan`: Container for complete layout information
  - Entity placements (combinators, poles, lamps, etc.)
  - Wire connections (red/green wires)
  - Tile placements
  - Entity configurations

- `EntityPlacement`: Represents a placed entity
  - Position (x, y)
  - Entity type
  - Configuration (control behavior, filters, etc.)

### entity_placer.py
**Combinator Placement**
- `EntityPlacer`: Places combinators for IR operations
  - Arithmetic → Arithmetic combinator
  - Decider → Decider combinator
  - Constant → Constant combinator
  - Memory → Memory cell (combinator pair)
  - Determines positions using grid-based placement

### signal_analyzer.py
**Signal Flow Analysis**
- `SignalAnalyzer`: Analyzes IR to understand signal dependencies
  - Builds signal usage graph
  - Tracks producers and consumers
  - Identifies signal lifetimes
  - Determines which signals need wires

### signal_graph.py
**Dependency Graph**
- `SignalGraph`: Represents signal dependencies
  - Nodes: IR operations
  - Edges: Signal flow between operations
  - Used for wire routing and optimization

### connection_planner.py
**Wire Connection Planning**
- `ConnectionPlanner`: Determines which entities to wire together
  - Analyzes signal flow from `signal_graph`
  - Minimizes wire count
  - Respects Factorio's wire distance limits

### wire_router.py
**Physical Wire Routing**
- `WireRouter`: Creates actual wire connections
  - Minimum Spanning Tree (MST) for efficient wiring
  - Handles red/green wire allocation
  - Avoids wire conflicts

### memory_builder.py
**Memory Cell Construction**
- `MemoryBuilder`: Creates memory cells from IR memory operations
  - SR/RS latches using decider combinators
  - Feedback loops for state storage
  - Read/write port configuration

### power_planner.py
**Power Pole Placement**
- `PowerPlanner`: Places power poles to cover all entities
  - Grid-based pole placement
  - Coverage radius calculation
  - Supports small/medium/large/substation poles

### tile_grid.py
**2D Grid Management**
- `TileGrid`: Manages spatial layout
  - Collision detection
  - Entity bounding boxes
  - Space allocation

### integer_layout_solver.py
**Layout Constraint Solving**
- Integer optimization for compact layouts
  - Minimize bounding box
  - Respect spacing constraints

## Information Flow

```
IR Operations (from lowering)
    ↓
[signal_analyzer.py] - Analyze signal flow
    ↓
Signal usage info + dependency graph
    ↓
[entity_placer.py] - Place combinators
    ↓
Entity positions (preliminary)
    ↓
[connection_planner.py] - Plan which entities to wire
    ↓
Wire connection list
    ↓
[wire_router.py] - Route physical wires (MST)
    ↓
Red/green wire connections
    ↓
[memory_builder.py] - Build memory cells
    ↓
Memory cell entity configurations
    ↓
[power_planner.py] - Place power poles
    ↓
Power pole placements
    ↓
[layout_plan.py] - Assemble final layout
    ↓
Complete LayoutPlan
    ↓
→ emission/emitter.py (generate blueprint string)
```

## Key Features

### Grid-Based Placement
Entities placed on a grid with spacing:
- Combinators: 2x1 tiles
- Memory cells: 2x2 (two combinators side-by-side)
- Power poles: various sizes

### Signal Flow Optimization
Analyzes IR to minimize wire connections:
- Common signals shared on wires
- Unused intermediate signals eliminated
- Wire merges optimized

### Memory Cell Design
Memory cells are combinator pairs in feedback loop:
```
[Decider 1] --red--> [Decider 2]
     ^                    |
     |----green wire------|
     (feedback loop for state)
```

### Power Coverage
Power planner ensures all entities powered:
- Calculates coverage radius
- Places poles to cover all entities
- Minimizes pole count

### Wire Distance Limits
Respects Factorio's wire distance limits (usually 7-9 tiles):
- Routes wires within range
- Adds intermediate relay poles if needed

## Usage Example

```python
from layout.planner import LayoutPlanner
from lowering.lowerer import ASTLowerer
# ... (parse, analyze, lower code)

# Plan layout
signal_refs = lowerer.signal_refs
referenced_signals = lowerer.referenced_signal_names

planner = LayoutPlanner(
    signal_type_map=lowerer.ir_builder.signal_type_map,
    diagnostics=diagnostics,
    signal_refs=signal_refs,
    referenced_signal_names=referenced_signals
)

layout_plan = planner.plan_layout(ir_operations)

# layout_plan contains:
# - entity_placements: list of EntityPlacement
# - connections: list of wire connections
# - tiles: list of tile placements
```

## Layout Algorithms

### Minimum Spanning Tree (MST)
Used for efficient wire routing:
- Kruskal's algorithm
- Minimizes total wire length
- Ensures all entities connected

### Greedy Placement
Entities placed greedily left-to-right, top-to-bottom:
- Simple and predictable
- Good for human readability

### Constraint Solving
Integer programming for compact layouts:
- Minimize bounding box area
- Respect spacing constraints
- Optional (can use greedy fallback)

## Testing

Tests verify:
- Entity placement correctness
- Wire routing validity
- Power coverage
- Memory cell construction
- Signal flow analysis
- Layout compactness

## Performance

- Most layouts planned in <100ms
- Scales linearly with IR operation count
- Grid-based algorithms are O(n)
- MST is O(n log n)

## Future Extensions

- More sophisticated placement algorithms
- Better handling of large bundles
- Compact memory cell designs
- Layout aesthetics (alignment, symmetry)
