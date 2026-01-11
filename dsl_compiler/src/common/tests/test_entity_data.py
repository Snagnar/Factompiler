"""Tests for entity_data.py - entity footprint and alignment utilities."""

from dsl_compiler.src.common.entity_data import (
    EntityDataHelper,
    get_entity_alignment,
    get_entity_footprint,
)


class TestEntityDataHelperGetFootprint:
    """Tests for EntityDataHelper.get_footprint() method."""

    def test_get_footprint_constant_combinator(self):
        """Test footprint for constant-combinator (1x1 in Factorio 2.0)."""
        width, height = EntityDataHelper.get_footprint("constant-combinator")
        assert width == 1
        assert height == 1

    def test_get_footprint_arithmetic_combinator(self):
        """Test footprint for arithmetic-combinator (1x2)."""
        width, height = EntityDataHelper.get_footprint("arithmetic-combinator")
        assert width == 1
        assert height == 2

    def test_get_footprint_decider_combinator(self):
        """Test footprint for decider-combinator (1x2)."""
        width, height = EntityDataHelper.get_footprint("decider-combinator")
        assert width == 1
        assert height == 2

    def test_get_footprint_small_lamp(self):
        """Test footprint for small-lamp (1x1)."""
        width, height = EntityDataHelper.get_footprint("small-lamp")
        assert width == 1
        assert height == 1

    def test_get_footprint_small_electric_pole(self):
        """Test footprint for small-electric-pole (1x1)."""
        width, height = EntityDataHelper.get_footprint("small-electric-pole")
        assert width == 1
        assert height == 1

    def test_get_footprint_medium_electric_pole(self):
        """Test footprint for medium-electric-pole (1x1)."""
        width, height = EntityDataHelper.get_footprint("medium-electric-pole")
        assert width == 1
        assert height == 1

    def test_get_footprint_substation(self):
        """Test footprint for substation (2x2)."""
        width, height = EntityDataHelper.get_footprint("substation")
        assert width == 2
        assert height == 2

    def test_get_footprint_unknown_entity_returns_default(self):
        """Test that unknown entities return default (1, 1) footprint."""
        width, height = EntityDataHelper.get_footprint("nonexistent-entity-xyz")
        assert width == 1
        assert height == 1

    def test_get_footprint_empty_string_returns_default(self):
        """Test that empty string returns default (1, 1) footprint."""
        width, height = EntityDataHelper.get_footprint("")
        assert width == 1
        assert height == 1

    def test_get_footprint_inserter(self):
        """Test footprint for inserter (1x1)."""
        width, height = EntityDataHelper.get_footprint("inserter")
        assert width == 1
        assert height == 1


class TestEntityDataHelperGetAlignment:
    """Tests for EntityDataHelper.get_alignment() method."""

    def test_get_alignment_1x1_entity(self):
        """Test alignment for 1x1 entities (should be 1)."""
        assert EntityDataHelper.get_alignment("small-lamp") == 1
        assert EntityDataHelper.get_alignment("small-electric-pole") == 1
        assert EntityDataHelper.get_alignment("inserter") == 1

    def test_get_alignment_1x2_entity(self):
        """Test alignment for 1x2 entities (should be 2 due to height)."""
        assert EntityDataHelper.get_alignment("arithmetic-combinator") == 2
        assert EntityDataHelper.get_alignment("decider-combinator") == 2

    def test_get_alignment_1x1_combinator_entity(self):
        """Test alignment for 1x1 constant-combinator (Factorio 2.0)."""
        assert EntityDataHelper.get_alignment("constant-combinator") == 1

    def test_get_alignment_2x2_entity(self):
        """Test alignment for 2x2 entities (should be 2)."""
        assert EntityDataHelper.get_alignment("substation") == 2

    def test_get_alignment_unknown_entity(self):
        """Test alignment for unknown entity returns 1 (default footprint 1x1)."""
        assert EntityDataHelper.get_alignment("nonexistent-entity-xyz") == 1


class TestModuleLevelFunctions:
    """Tests for module-level convenience functions."""

    def test_get_entity_footprint_is_alias(self):
        """Test that get_entity_footprint is an alias for EntityDataHelper.get_footprint."""
        assert get_entity_footprint("constant-combinator") == EntityDataHelper.get_footprint(
            "constant-combinator"
        )

    def test_get_entity_alignment_is_alias(self):
        """Test that get_entity_alignment is an alias for EntityDataHelper.get_alignment."""
        assert get_entity_alignment("constant-combinator") == EntityDataHelper.get_alignment(
            "constant-combinator"
        )

    def test_module_functions_work_correctly(self):
        """Test module-level functions work correctly."""
        assert get_entity_footprint("small-lamp") == (1, 1)
        assert get_entity_alignment("small-lamp") == 1
        assert get_entity_footprint("arithmetic-combinator") == (1, 2)
        assert get_entity_alignment("arithmetic-combinator") == 2


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_get_footprint_with_tile_width_height(self):
        """Test entities that have explicit tile_width and tile_height."""
        from unittest.mock import patch

        # Mock draftsman data to have an entity with explicit tile dimensions
        mock_raw = {
            "mock-entity": {
                "tile_width": 3,
                "tile_height": 4,
            }
        }

        with patch("dsl_compiler.src.common.entity_data.entity_data") as mock_data:
            mock_data.raw.get.return_value = mock_raw["mock-entity"]
            result = EntityDataHelper.get_footprint("mock-entity")
            assert result == (3, 4)

    def test_get_footprint_exception_handling(self):
        """Test that exceptions return default (1, 1)."""
        from unittest.mock import patch

        with patch("dsl_compiler.src.common.entity_data.entity_data") as mock_data:
            mock_data.raw.get.side_effect = Exception("Simulated error")
            result = EntityDataHelper.get_footprint("any-entity")
            assert result == (1, 1)
