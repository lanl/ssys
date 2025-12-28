import json

from bioservices import BioModels

bm = BioModels()
test_ids = ["BIOMD0000000001", "MODEL1910120001"]

for model_id in test_ids:
    print(f"\n{'=' * 60}")
    print(f"Model: {model_id}")
    print("=" * 60)
    try:
        info = bm.get_model(model_id)
        if info:
            # Show structure
            if isinstance(info, dict):
                print("Keys:", list(info.keys()))
                if "format" in info:
                    print("format:", info["format"])
                if "submitter" in info:
                    print("submitter:", info["submitter"])
            print("\nFull response:")
            print(json.dumps(info, indent=2, default=str)[:500])
        else:
            print("Returned None")
    except Exception as e:
        print(f"ERROR: {e}")
