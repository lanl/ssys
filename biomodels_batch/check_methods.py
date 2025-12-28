from bioservices import BioModels

bm = BioModels()
print("Available methods:")
for attr in dir(bm):
    if not attr.startswith("_") and "model" in attr.lower():
        print(f"  - {attr}")
