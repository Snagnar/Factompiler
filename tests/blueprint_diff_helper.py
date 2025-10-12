"""
Blueprint diff helper: counts entity types in a blueprint for rapid verification.
"""

from draftsman.blueprintable import Blueprint
from collections import Counter


def count_entity_types(blueprint: Blueprint):
    """Return a Counter of entity types in the blueprint."""
    types = [getattr(e, "name", None) for e in blueprint.entities]
    return Counter(types)


# Example usage:
# blueprint = ...
# print(count_entity_types(blueprint))
