"""
Tests for layout/tile_grid.py - Tile occupancy tracking.
"""

from dsl_compiler.src.layout.layout_plan import EntityPlacement
from dsl_compiler.src.layout.tile_grid import TileGrid


class TestTileGridInit:
    """Tests for TileGrid initialization."""

    def test_init_creates_empty_grid(self):
        """Test TileGrid initializes with no occupied tiles."""
        grid = TileGrid()
        assert grid is not None


class TestTileGridIsAvailable:
    """Tests for TileGrid.is_available method."""

    def test_empty_grid_is_available(self):
        """Test empty grid has all tiles available."""
        grid = TileGrid()
        assert grid.is_available((0, 0), (1, 1)) is True
        assert grid.is_available((10, 10), (1, 1)) is True

    def test_single_tile_availability(self):
        """Test availability of single tile footprint."""
        grid = TileGrid()
        grid.mark_occupied((5, 5), (1, 1))
        assert grid.is_available((5, 5), (1, 1)) is False
        assert grid.is_available((4, 5), (1, 1)) is True
        assert grid.is_available((6, 5), (1, 1)) is True

    def test_multi_tile_availability(self):
        """Test availability check for multi-tile footprint."""
        grid = TileGrid()
        grid.mark_occupied((5, 5), (2, 2))
        # All occupied tiles should be unavailable
        assert grid.is_available((5, 5), (1, 1)) is False
        assert grid.is_available((6, 5), (1, 1)) is False
        assert grid.is_available((5, 6), (1, 1)) is False
        assert grid.is_available((6, 6), (1, 1)) is False
        # Adjacent tiles should be available
        assert grid.is_available((4, 5), (1, 1)) is True
        assert grid.is_available((7, 5), (1, 1)) is True

    def test_footprint_overlap_check(self):
        """Test that footprint overlap is detected."""
        grid = TileGrid()
        grid.mark_occupied((5, 5), (1, 1))
        # 2x2 footprint at (4, 4) would occupy (4,4), (5,4), (4,5), (5,5)
        # (5,5) is occupied, so should not be available
        assert grid.is_available((4, 4), (2, 2)) is False
        # 2x2 at (6,5) should be fine
        assert grid.is_available((6, 5), (2, 2)) is True


class TestTileGridMarkOccupied:
    """Tests for TileGrid.mark_occupied method."""

    def test_mark_single_tile(self):
        """Test marking a single tile."""
        grid = TileGrid()
        grid.mark_occupied((3, 3), (1, 1))
        assert grid.is_available((3, 3), (1, 1)) is False

    def test_mark_multi_tile(self):
        """Test marking a multi-tile footprint."""
        grid = TileGrid()
        grid.mark_occupied((0, 0), (3, 2))
        # Check all 6 tiles are occupied
        for x in range(3):
            for y in range(2):
                assert grid.is_available((x, y), (1, 1)) is False
        # Outside the footprint should be available
        assert grid.is_available((3, 0), (1, 1)) is True
        assert grid.is_available((0, 2), (1, 1)) is True

    def test_mark_occupied_twice(self):
        """Test marking same tile twice doesn't cause issues."""
        grid = TileGrid()
        grid.mark_occupied((1, 1), (1, 1))
        grid.mark_occupied((1, 1), (1, 1))  # Mark again
        assert grid.is_available((1, 1), (1, 1)) is False


class TestTileGridReserveExact:
    """Tests for TileGrid.reserve_exact method."""

    def test_reserve_available_position(self):
        """Test reserving an available position succeeds."""
        grid = TileGrid()
        result = grid.reserve_exact((5, 5), (1, 1))
        assert result is True
        assert grid.is_available((5, 5), (1, 1)) is False

    def test_reserve_unavailable_position(self):
        """Test reserving an unavailable position fails."""
        grid = TileGrid()
        grid.mark_occupied((5, 5), (1, 1))
        result = grid.reserve_exact((5, 5), (1, 1))
        assert result is False

    def test_reserve_multi_tile_available(self):
        """Test reserving multi-tile footprint when available."""
        grid = TileGrid()
        result = grid.reserve_exact((0, 0), (2, 2))
        assert result is True
        # All tiles should now be occupied
        assert grid.is_available((0, 0), (1, 1)) is False
        assert grid.is_available((1, 1), (1, 1)) is False

    def test_reserve_multi_tile_partial_overlap(self):
        """Test reserving multi-tile footprint with partial overlap fails."""
        grid = TileGrid()
        grid.mark_occupied((1, 1), (1, 1))
        # Try to reserve 2x2 at (0,0) which would overlap (1,1)
        result = grid.reserve_exact((0, 0), (2, 2))
        assert result is False
        # (0,0) should still be available since reservation failed
        assert grid.is_available((0, 0), (1, 1)) is True


class TestTileGridRebuildFromPlacements:
    """Tests for TileGrid.rebuild_from_placements method."""

    def test_rebuild_empty_placements(self):
        """Test rebuilding from empty placements."""
        grid = TileGrid()
        grid.mark_occupied((5, 5), (1, 1))  # Pre-mark something
        grid.rebuild_from_placements({})
        # Should be cleared
        assert grid.is_available((5, 5), (1, 1)) is True

    def test_rebuild_single_placement(self):
        """Test rebuilding from single placement."""
        grid = TileGrid()
        placements = {
            "entity-1": EntityPlacement(
                ir_node_id="entity-1",
                entity_type="arithmetic-combinator",
                position=(1.5, 1.0),  # Center position
                properties={"footprint": (1, 2)},
            )
        }
        grid.rebuild_from_placements(placements)
        # Footprint 1x2 centered at (1.5, 1.0):
        # tile_x = int(1.5 - 1/2.0) = int(1.0) = 1
        # tile_y = int(1.0 - 2/2.0) = int(0.0) = 0
        # Occupies tiles (1, 0) and (1, 1)
        assert grid.is_available((1, 0), (1, 1)) is False
        assert grid.is_available((1, 1), (1, 1)) is False
        # Adjacent tiles available
        assert grid.is_available((0, 0), (1, 1)) is True
        assert grid.is_available((2, 0), (1, 1)) is True

    def test_rebuild_multiple_placements(self):
        """Test rebuilding from multiple placements."""
        grid = TileGrid()
        placements = {
            "entity-1": EntityPlacement(
                ir_node_id="entity-1",
                entity_type="constant-combinator",
                position=(0.5, 0.5),
                properties={"footprint": (1, 1)},
            ),
            "entity-2": EntityPlacement(
                ir_node_id="entity-2",
                entity_type="constant-combinator",
                position=(5.5, 5.5),
                properties={"footprint": (1, 1)},
            ),
        }
        grid.rebuild_from_placements(placements)
        assert grid.is_available((0, 0), (1, 1)) is False
        assert grid.is_available((5, 5), (1, 1)) is False
        # Other tiles available
        assert grid.is_available((2, 2), (1, 1)) is True

    def test_rebuild_placement_without_position(self):
        """Test rebuilding skips placements without position."""
        grid = TileGrid()
        placements = {
            "entity-1": EntityPlacement(
                ir_node_id="entity-1",
                entity_type="constant-combinator",
                position=None,  # No position
                properties={"footprint": (1, 1)},
            ),
        }
        grid.rebuild_from_placements(placements)
        # Should not raise, no tiles marked
        assert grid.is_available((0, 0), (1, 1)) is True

    def test_rebuild_placement_default_footprint(self):
        """Test rebuilding uses default 1x1 footprint if not specified."""
        grid = TileGrid()
        placements = {
            "entity-1": EntityPlacement(
                ir_node_id="entity-1",
                entity_type="small-lamp",
                position=(0.5, 0.5),
                properties={},  # No footprint specified
            ),
        }
        grid.rebuild_from_placements(placements)
        assert grid.is_available((0, 0), (1, 1)) is False

    def test_rebuild_clears_previous_occupancy(self):
        """Test rebuilding clears previous occupancy."""
        grid = TileGrid()
        grid.mark_occupied((10, 10), (1, 1))
        placements = {
            "entity-1": EntityPlacement(
                ir_node_id="entity-1",
                entity_type="constant-combinator",
                position=(0.5, 0.5),
                properties={"footprint": (1, 1)},
            ),
        }
        grid.rebuild_from_placements(placements)
        # Previous occupancy should be cleared
        assert grid.is_available((10, 10), (1, 1)) is True
        # New occupancy from placement
        assert grid.is_available((0, 0), (1, 1)) is False
