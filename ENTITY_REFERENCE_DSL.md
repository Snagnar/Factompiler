# Entity Properties for Circuit DSL

This reference shows entity properties relevant for circuit control in your DSL.
Generated for Draftsman 3.2.0

## Quick Links

- [Draftsman ReadTheDocs](https://factorio-draftsman.readthedocs.io/en/latest/)
- [Draftsman GitHub](https://github.com/redruin1/factorio-draftsman)

## Combinators

### ArithmeticCombinator

**Circuit/Dynamic Properties:**

| Property | Type |
|----------|------|
| `direction` | Direction |
| `first_operand` | NoneType |
| `item_requests` | list |
| `name` | str |
| `operation` | str |
| `output_signal` | NoneType |
| `player_description` | str |
| `position` | _PosVector |
| `quality` | str |
| `second_operand` | int |
| `tile_position` | _TileVector |

**Static Properties (set at placement):**

- `first_operand_wires` (CircuitNetworkSelection)
- `second_operand_wires` (CircuitNetworkSelection)


### DeciderCombinator

**Circuit/Dynamic Properties:**

| Property | Type |
|----------|------|
| `conditions` | list |
| `direction` | Direction |
| `item_requests` | list |
| `name` | str |
| `outputs` | list |
| `player_description` | str |
| `position` | _PosVector |
| `quality` | str |
| `tile_position` | _TileVector |


### ConstantCombinator

**Circuit/Dynamic Properties:**

| Property | Type |
|----------|------|
| `direction` | Direction |
| `enabled` | bool |
| `item_requests` | list |
| `name` | str |
| `player_description` | str |
| `position` | _PosVector |
| `quality` | str |
| `sections` | list |
| `tile_position` | _TileVector |


### SelectorCombinator

**Circuit/Dynamic Properties:**

| Property | Type |
|----------|------|
| `count_signal` | NoneType |
| `direction` | Direction |
| `index_signal` | NoneType |
| `item_requests` | list |
| `name` | str |
| `operation` | str |
| `player_description` | str |
| `position` | _PosVector |
| `quality` | str |
| `quality_destination_signal` | NoneType |
| `quality_source_signal` | NoneType |
| `select_quality_from_signal` | bool |
| `tile_position` | _TileVector |

**Static Properties (set at placement):**

- `index_constant` (int)
- `quality_filter` (QualityFilter)
- `quality_source_static` (str)
- `random_update_interval` (int)
- `select_max` (bool)


## Lamps & Displays

### Lamp

**Circuit/Dynamic Properties:**

| Property | Type |
|----------|------|
| `always_on` | bool |
| `blue_signal` | SignalID |
| `circuit_condition` | Condition |
| `circuit_enabled` | bool |
| `color` | Color |
| `color_mode` | LampColorMode |
| `connect_to_logistic_network` | bool |
| `green_signal` | SignalID |
| `item_requests` | list |
| `logistic_condition` | Condition |
| `name` | str |
| `position` | _PosVector |
| `quality` | str |
| `red_signal` | SignalID |
| `rgb_signal` | SignalID |
| `tile_position` | _TileVector |
| `use_colors` | bool |


### DisplayPanel

**Circuit/Dynamic Properties:**

| Property | Type |
|----------|------|
| `direction` | Direction |
| `item_requests` | list |
| `name` | str |
| `player_description` | str |
| `position` | _PosVector |
| `quality` | str |
| `tile_position` | _TileVector |

**Static Properties (set at placement):**

- `always_show_in_alt_mode` (bool)
- `icon` (NoneType)
- `messages` (list)
- `show_in_chart` (bool)
- `text` (str)


## Inserters

### Inserter

**Circuit/Dynamic Properties:**

| Property | Type |
|----------|------|
| `circuit_condition` | Condition |
| `circuit_enabled` | bool |
| `circuit_set_filters` | bool |
| `circuit_set_stack_size` | bool |
| `connect_to_logistic_network` | bool |
| `direction` | Direction |
| `filter_mode` | str |
| `filters` | list |
| `item_requests` | list |
| `logistic_condition` | Condition |
| `name` | str |
| `override_stack_size` | NoneType |
| `position` | _PosVector |
| `quality` | str |
| `stack_size_control_signal` | NoneType |
| `tile_position` | _TileVector |
| `read_hand_contents` | bool |
| `read_mode` | InserterReadMode |

**Static Properties (set at placement):**

- `drop_position_offset` (Vector)
- `mode_of_operation` (InserterModeOfOperation)
- `pickup_position_offset` (Vector)
- `spoil_priority` (NoneType)
- `use_filters` (bool)


## Belts

### TransportBelt

**Circuit/Dynamic Properties:**

| Property | Type |
|----------|------|
| `circuit_condition` | Condition |
| `circuit_enabled` | bool |
| `connect_to_logistic_network` | bool |
| `direction` | Direction |
| `item_requests` | list |
| `logistic_condition` | Condition |
| `name` | str |
| `position` | _PosVector |
| `quality` | str |
| `tile_position` | _TileVector |
| `read_contents` | bool |
| `read_mode` | BeltReadMode |


### UndergroundBelt

**Circuit/Dynamic Properties:**

| Property | Type |
|----------|------|
| `direction` | Direction |
| `io_type` | str |
| `item_requests` | list |
| `name` | str |
| `position` | _PosVector |
| `quality` | str |
| `tile_position` | _TileVector |


### Splitter

**Circuit/Dynamic Properties:**

| Property | Type |
|----------|------|
| `direction` | Direction |
| `item_requests` | list |
| `name` | str |
| `position` | _PosVector |
| `quality` | str |
| `tile_position` | _TileVector |
| `input_left_condition` | Condition |
| `input_right_condition` | Condition |
| `output_left_condition` | Condition |
| `output_right_condition` | Condition |

**Static Properties (set at placement):**

- `filter` (NoneType)
- `input_priority` (str)
- `output_priority` (str)
- `set_filter` (bool)
- `set_input_side` (bool)
- `set_output_side` (bool)


### Loader

**Circuit/Dynamic Properties:**

| Property | Type |
|----------|------|
| `direction` | Direction |
| `filters` | list |
| `io_type` | str |
| `item_requests` | list |
| `name` | str |
| `position` | _PosVector |
| `quality` | str |
| `tile_position` | _TileVector |

**Static Properties (set at placement):**

- `use_filters` (bool)


## Train System

### TrainStop

**Circuit/Dynamic Properties:**

| Property | Type |
|----------|------|
| `circuit_condition` | Condition |
| `circuit_enabled` | bool |
| `color` | Color |
| `connect_to_logistic_network` | bool |
| `direction` | Direction |
| `item_requests` | list |
| `logistic_condition` | Condition |
| `manual_trains_limit` | NoneType |
| `name` | str |
| `position` | _PosVector |
| `priority` | int |
| `priority_signal` | SignalID |
| `quality` | str |
| `read_from_train` | bool |
| `read_stopped_train` | bool |
| `read_trains_count` | bool |
| `send_to_train` | bool |
| `set_priority` | bool |
| `signal_limits_trains` | bool |
| `station` | str |
| `tile_position` | _TileVector |
| `train_stopped_signal` | SignalID |
| `trains_count_signal` | SignalID |
| `trains_limit_signal` | SignalID |


### RailSignal

**Circuit/Dynamic Properties:**

| Property | Type |
|----------|------|
| `circuit_condition` | Condition |
| `circuit_enabled` | bool |
| `direction` | Direction |
| `green_output_signal` | SignalID |
| `item_requests` | list |
| `name` | str |
| `position` | _PosVector |
| `quality` | str |
| `read_signal` | bool |
| `red_output_signal` | SignalID |
| `tile_position` | _TileVector |
| `yellow_output_signal` | SignalID |


### RailChainSignal

**Circuit/Dynamic Properties:**

| Property | Type |
|----------|------|
| `blue_output_signal` | SignalID |
| `direction` | Direction |
| `green_output_signal` | SignalID |
| `item_requests` | list |
| `name` | str |
| `position` | _PosVector |
| `quality` | str |
| `red_output_signal` | SignalID |
| `tile_position` | _TileVector |
| `yellow_output_signal` | SignalID |


### Locomotive

**Circuit/Dynamic Properties:**

| Property | Type |
|----------|------|
| `color` | Color |
| `item_requests` | list |
| `name` | str |
| `position` | _PosVector |
| `quality` | str |
| `tile_position` | _TileVector |
| `enable_logistics_while_moving` | bool |

**Static Properties (set at placement):**

- `equipment` (list)
- `orientation` (Orientation)


### CargoWagon

**Circuit/Dynamic Properties:**

| Property | Type |
|----------|------|
| `item_requests` | list |
| `name` | str |
| `position` | _PosVector |
| `quality` | str |
| `tile_position` | _TileVector |
| `enable_logistics_while_moving` | bool |

**Static Properties (set at placement):**

- `equipment` (list)
- `inventory` (Inventory)
- `orientation` (Orientation)


### FluidWagon

**Circuit/Dynamic Properties:**

| Property | Type |
|----------|------|
| `item_requests` | list |
| `name` | str |
| `position` | _PosVector |
| `quality` | str |
| `tile_position` | _TileVector |
| `enable_logistics_while_moving` | bool |

**Static Properties (set at placement):**

- `equipment` (list)
- `orientation` (Orientation)


## Production

### AssemblingMachine

**Circuit/Dynamic Properties:**

| Property | Type |
|----------|------|
| `circuit_condition` | Condition |
| `circuit_enabled` | bool |
| `circuit_set_recipe` | bool |
| `connect_to_logistic_network` | bool |
| `direction` | Direction |
| `item_requests` | list |
| `logistic_condition` | Condition |
| `name` | str |
| `position` | _PosVector |
| `quality` | str |
| `recipe` | NoneType |
| `recipe_finished_signal` | NoneType |
| `tile_position` | _TileVector |
| `working_signal` | NoneType |
| `read_contents` | bool |
| `read_recipe_finished` | bool |
| `read_working` | bool |

**Static Properties (set at placement):**

- `include_in_crafting` (bool)
- `recipe_quality` (str)


### Furnace

**Circuit/Dynamic Properties:**

| Property | Type |
|----------|------|
| `circuit_condition` | Condition |
| `circuit_enabled` | bool |
| `circuit_set_recipe` | bool |
| `connect_to_logistic_network` | bool |
| `direction` | Direction |
| `item_requests` | list |
| `logistic_condition` | Condition |
| `name` | str |
| `position` | _PosVector |
| `quality` | str |
| `recipe_finished_signal` | NoneType |
| `tile_position` | _TileVector |
| `working_signal` | NoneType |
| `read_contents` | bool |
| `read_recipe_finished` | bool |
| `read_working` | bool |

**Static Properties (set at placement):**

- `include_in_crafting` (bool)


### MiningDrill

**Circuit/Dynamic Properties:**

| Property | Type |
|----------|------|
| `circuit_condition` | Condition |
| `circuit_enabled` | bool |
| `connect_to_logistic_network` | bool |
| `direction` | Direction |
| `item_requests` | list |
| `logistic_condition` | Condition |
| `name` | str |
| `position` | _PosVector |
| `quality` | str |
| `tile_position` | _TileVector |
| `read_mode` | MiningDrillReadMode |
| `read_resources` | bool |


### Lab

**Circuit/Dynamic Properties:**

| Property | Type |
|----------|------|
| `item_requests` | list |
| `name` | str |
| `position` | _PosVector |
| `quality` | str |
| `tile_position` | _TileVector |


## Storage

### Container

**Circuit/Dynamic Properties:**

| Property | Type |
|----------|------|
| `bar` | NoneType |
| `item_requests` | list |
| `name` | str |
| `position` | _PosVector |
| `quality` | str |
| `tile_position` | _TileVector |


### LogisticPassiveContainer

**Circuit/Dynamic Properties:**

| Property | Type |
|----------|------|
| `bar` | NoneType |
| `circuit_condition` | Condition |
| `circuit_enabled` | bool |
| `item_requests` | list |
| `name` | str |
| `position` | _PosVector |
| `quality` | str |
| `tile_position` | _TileVector |
| `read_contents` | bool |


### LogisticActiveContainer

**Circuit/Dynamic Properties:**

| Property | Type |
|----------|------|
| `bar` | NoneType |
| `circuit_condition` | Condition |
| `item_requests` | list |
| `name` | str |
| `position` | _PosVector |
| `quality` | str |
| `tile_position` | _TileVector |
| `read_contents` | bool |


### LogisticStorageContainer

**Circuit/Dynamic Properties:**

| Property | Type |
|----------|------|
| `bar` | NoneType |
| `circuit_condition` | Condition |
| `circuit_enabled` | bool |
| `item_requests` | list |
| `name` | str |
| `position` | _PosVector |
| `quality` | str |
| `sections` | list |
| `tile_position` | _TileVector |
| `read_contents` | bool |
| `requests_enabled` | bool |

**Static Properties (set at placement):**

- `request_from_buffers` (bool)
- `trash_not_requested` (bool)


### LogisticRequestContainer

**Circuit/Dynamic Properties:**

| Property | Type |
|----------|------|
| `bar` | NoneType |
| `circuit_condition` | Condition |
| `circuit_enabled` | bool |
| `item_requests` | list |
| `name` | str |
| `position` | _PosVector |
| `quality` | str |
| `sections` | list |
| `tile_position` | _TileVector |
| `requests_enabled` | bool |

**Static Properties (set at placement):**

- `mode_of_operation` (LogisticModeOfOperation)
- `request_from_buffers` (bool)
- `trash_not_requested` (bool)


### LogisticBufferContainer

**Circuit/Dynamic Properties:**

| Property | Type |
|----------|------|
| `bar` | NoneType |
| `circuit_condition` | Condition |
| `circuit_enabled` | bool |
| `item_requests` | list |
| `name` | str |
| `position` | _PosVector |
| `quality` | str |
| `sections` | list |
| `tile_position` | _TileVector |
| `requests_enabled` | bool |

**Static Properties (set at placement):**

- `mode_of_operation` (LogisticModeOfOperation)
- `request_from_buffers` (bool)
- `trash_not_requested` (bool)


## Power

### ElectricPole

**Circuit/Dynamic Properties:**

| Property | Type |
|----------|------|
| `item_requests` | list |
| `name` | str |
| `position` | _PosVector |
| `quality` | str |
| `tile_position` | _TileVector |


### PowerSwitch

**Circuit/Dynamic Properties:**

| Property | Type |
|----------|------|
| `circuit_condition` | Condition |
| `connect_to_logistic_network` | bool |
| `item_requests` | list |
| `logistic_condition` | Condition |
| `name` | str |
| `position` | _PosVector |
| `quality` | str |
| `tile_position` | _TileVector |

**Static Properties (set at placement):**

- `switch_state` (bool)


### Accumulator

**Circuit/Dynamic Properties:**

| Property | Type |
|----------|------|
| `item_requests` | list |
| `name` | str |
| `output_signal` | SignalID |
| `position` | _PosVector |
| `quality` | str |
| `tile_position` | _TileVector |


## Fluid

### Pump

**Circuit/Dynamic Properties:**

| Property | Type |
|----------|------|
| `circuit_condition` | Condition |
| `connect_to_logistic_network` | bool |
| `direction` | Direction |
| `item_requests` | list |
| `logistic_condition` | Condition |
| `name` | str |
| `position` | _PosVector |
| `quality` | str |
| `tile_position` | _TileVector |


### StorageTank

**Circuit/Dynamic Properties:**

| Property | Type |
|----------|------|
| `direction` | Direction |
| `item_requests` | list |
| `name` | str |
| `position` | _PosVector |
| `quality` | str |
| `tile_position` | _TileVector |


### OffshorePump

**Circuit/Dynamic Properties:**

| Property | Type |
|----------|------|
| `circuit_condition` | Condition |
| `connect_to_logistic_network` | bool |
| `direction` | Direction |
| `item_requests` | list |
| `logistic_condition` | Condition |
| `name` | str |
| `position` | _PosVector |
| `quality` | str |
| `tile_position` | _TileVector |


## Other

### ProgrammableSpeaker

**Circuit/Dynamic Properties:**

| Property | Type |
|----------|------|
| `circuit_condition` | Condition |
| `circuit_enabled` | bool |
| `item_requests` | list |
| `name` | str |
| `position` | _PosVector |
| `quality` | str |
| `signal_value_is_pitch` | bool |
| `tile_position` | _TileVector |
| `volume_controlled_by_signal` | bool |
| `volume_signal` | NoneType |

**Static Properties (set at placement):**

- `alert_icon` (NoneType)
- `alert_message` (str)
- `allow_polyphony` (bool)
- `instrument_id` (int)
- `instrument_name` (str)
- `note_id` (int)
- `note_name` (str)
- `playback_mode` (str)
- ... and 4 more


### Roboport

**Circuit/Dynamic Properties:**

| Property | Type |
|----------|------|
| `available_construction_signal` | SignalID |
| `available_logistic_signal` | SignalID |
| `item_requests` | list |
| `name` | str |
| `position` | _PosVector |
| `quality` | str |
| `roboport_count_signal` | SignalID |
| `sections` | list |
| `tile_position` | _TileVector |
| `total_construction_signal` | SignalID |
| `total_logistic_signal` | SignalID |
| `read_items_mode` | ReadItemsMode |
| `read_robot_stats` | bool |
| `requests_enabled` | bool |

**Static Properties (set at placement):**

- `request_from_buffers` (bool)
- `trash_not_requested` (bool)


### Radar

**Circuit/Dynamic Properties:**

| Property | Type |
|----------|------|
| `item_requests` | list |
| `name` | str |
| `position` | _PosVector |
| `quality` | str |
| `tile_position` | _TileVector |


---

## All Entity Classes

Complete list of all 87 entity classes in Draftsman:

- [Accumulator](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/accumulator.html)
- [AgriculturalTower](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/agricultural_tower.html)
- [AmmoTurret](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/ammo_turret.html)
- [ArithmeticCombinator](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/arithmetic_combinator.html)
- [ArtilleryTurret](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/artillery_turret.html)
- [ArtilleryWagon](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/artillery_wagon.html)
- [AssemblingMachine](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/assembling_machine.html)
- [AsteroidCollector](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/asteroid_collector.html)
- [Beacon](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/beacon.html)
- [Boiler](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/boiler.html)
- [BurnerGenerator](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/burner_generator.html)
- [Car](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/car.html)
- [CargoBay](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/cargo_bay.html)
- [CargoLandingPad](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/cargo_landing_pad.html)
- [CargoWagon](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/cargo_wagon.html)
- [ConstantCombinator](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/constant_combinator.html)
- [Container](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/container.html)
- [CurvedRailA](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/curved_rail_a.html)
- [CurvedRailB](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/curved_rail_b.html)
- [DeciderCombinator](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/decider_combinator.html)
- [DisplayPanel](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/display_panel.html)
- [ElectricEnergyInterface](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/electric_energy_interface.html)
- [ElectricPole](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/electric_pole.html)
- [ElectricTurret](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/electric_turret.html)
- [ElevatedCurvedRailA](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/elevated_curved_rail_a.html)
- [ElevatedCurvedRailB](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/elevated_curved_rail_b.html)
- [ElevatedHalfDiagonalRail](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/elevated_half_diagonal_rail.html)
- [ElevatedStraightRail](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/elevated_straight_rail.html)
- [FluidTurret](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/fluid_turret.html)
- [FluidWagon](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/fluid_wagon.html)
- [Furnace](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/furnace.html)
- [FusionGenerator](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/fusion_generator.html)
- [FusionReactor](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/fusion_reactor.html)
- [Gate](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/gate.html)
- [Generator](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/generator.html)
- [HalfDiagonalRail](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/half_diagonal_rail.html)
- [HeatInterface](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/heat_interface.html)
- [HeatPipe](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/heat_pipe.html)
- [InfinityContainer](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/infinity_container.html)
- [InfinityPipe](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/infinity_pipe.html)
- [Inserter](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/inserter.html)
- [Lab](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/lab.html)
- [Lamp](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/lamp.html)
- [LandMine](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/land_mine.html)
- [LegacyCurvedRail](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/legacy_curved_rail.html)
- [LegacyStraightRail](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/legacy_straight_rail.html)
- [LightningAttractor](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/lightning_attractor.html)
- [LinkedBelt](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/linked_belt.html)
- [LinkedContainer](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/linked_container.html)
- [Loader](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/loader.html)
- [Locomotive](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/locomotive.html)
- [LogisticActiveContainer](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/logistic_active_container.html)
- [LogisticBufferContainer](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/logistic_buffer_container.html)
- [LogisticPassiveContainer](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/logistic_passive_container.html)
- [LogisticRequestContainer](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/logistic_request_container.html)
- [LogisticStorageContainer](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/logistic_storage_container.html)
- [MiningDrill](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/mining_drill.html)
- [OffshorePump](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/offshore_pump.html)
- [Pipe](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/pipe.html)
- [PlayerPort](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/player_port.html)
- [PowerSwitch](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/power_switch.html)
- [ProgrammableSpeaker](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/programmable_speaker.html)
- [Pump](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/pump.html)
- [Radar](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/radar.html)
- [RailChainSignal](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/rail_chain_signal.html)
- [RailRamp](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/rail_ramp.html)
- [RailSignal](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/rail_signal.html)
- [RailSupport](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/rail_support.html)
- [Reactor](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/reactor.html)
- [Roboport](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/roboport.html)
- [RocketSilo](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/rocket_silo.html)
- [SelectorCombinator](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/selector_combinator.html)
- [SimpleEntityWithForce](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/simple_entity_with_force.html)
- [SimpleEntityWithOwner](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/simple_entity_with_owner.html)
- [SolarPanel](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/solar_panel.html)
- [SpacePlatformHub](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/space_platform_hub.html)
- [SpiderVehicle](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/spider_vehicle.html)
- [Splitter](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/splitter.html)
- [StorageTank](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/storage_tank.html)
- [StraightRail](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/straight_rail.html)
- [Thruster](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/thruster.html)
- [TrainStop](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/train_stop.html)
- [TransportBelt](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/transport_belt.html)
- [UndergroundBelt](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/underground_belt.html)
- [UndergroundPipe](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/underground_pipe.html)
- [Valve](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/valve.html)
- [Wall](https://factorio-draftsman.readthedocs.io/en/latest/reference/prototypes/wall.html)