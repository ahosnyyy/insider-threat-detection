#!/usr/bin/env python
"""Quick test for role_units mapping and FeatureExtractor."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.features import (
    load_role_units_mapping,
    FeatureExtractor,
    ROLE_FEATURES_NONE,
    ROLE_FEATURES_ROLES,
    ROLE_FEATURES_UNITS,
)

p = Path("config/role_units.yaml")
unit_order, role_to_unit, default_unit = load_role_units_mapping(p)
print("unit_order:", unit_order)
print("roles in Administration:", [r for r, u in role_to_unit.items() if u == "Administration"][:5], "...")
print("default_unit:", default_unit)

e = FeatureExtractor(role_features=ROLE_FEATURES_NONE)
print("none feature_dim (approx):", len(e.base_features))

e2 = FeatureExtractor(role_features=ROLE_FEATURES_UNITS, role_mapping_path=p)
print("units: role_categories (before fit):", e2.role_categories)

print("OK")
