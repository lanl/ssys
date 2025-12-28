from bioservices import BioModels

bm = BioModels()
# Test a few curated models
test_ids = [
    "BIOMD0000000001",
    "BIOMD0000000002",
    "BIOMD0000000003",
    "BIOMD0000000010",
    "BIOMD0000000020",
]

for model_id in test_ids:
    try:
        info = bm.get_model(model_id)
        if info and isinstance(info, dict):
            approach = info.get("modellingApproach", "N/A")
            fmt = info.get("format", {})
            print(f"{model_id}: modellingApproach={approach}, format={fmt.get('name', 'N/A')}")
    except Exception:
        pass
