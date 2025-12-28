#!/usr/bin/env python3
"""Debug script to inspect BioModels API responses."""

import json

from bioservices import BioModels

bm = BioModels()

# Test with a few known models
test_models = [
    "BIOMD0000000001",  # Known curated ODE model
    "BIOMD0000000010",  # Another curated model
    "MODEL1910120001",  # Non-curated model
]

for model_id in test_models:
    print(f"\n{'=' * 60}")
    print(f"Model: {model_id}")
    print("=" * 60)

    try:
        info = bm.get_model_by_id(model_id)
        if info:
            print(json.dumps(info, indent=2, default=str))
        else:
            print("ERROR: get_model_by_id() returned None")
    except Exception as e:
        print(f"ERROR: {e}")
