import json
from collections import Counter

# Step 1: Read the original JSON file
with open('rust_api_info.json', 'r') as f:
    data = json.load(f)

# Step 2: Filter APIs with stability "1.0.0"
filtered_apis = {key: value for key, value in data['apis'].items() if value['stability'] == "1.0.0"}

# Step 3: Compute updated statistics based on filtered APIs
total_apis = len(filtered_apis)

# Count APIs by type
by_type = Counter(value['type'] for value in filtered_apis.values())

# Since all filtered APIs have stability "1.0.0", update by_stability accordingly
by_stability = {"1.0.0": total_apis}

# Count deprecated APIs
deprecated_count = sum(1 for value in filtered_apis.values() if value['deprecated'])

# Count APIs with examples (at least one example)
with_examples = sum(1 for value in filtered_apis.values() if len(value['examples']) > 0)

# Total number of examples across all filtered APIs
total_examples = sum(len(value['examples']) for value in filtered_apis.values())

# Create new statistics dictionary
new_statistics = {
    "total_apis": total_apis,
    "by_type": dict(by_type),
    "by_stability": by_stability,
    "deprecated_count": deprecated_count,
    "with_examples": with_examples,
    "total_examples": total_examples
}

# Step 4: Construct the new JSON data
# new_data = {
#     "apis": filtered_apis
# }

new_data = filtered_apis

# Step 5: Write the new JSON to rust_api_info_stable_1.json
with open('../scripts/rust_api_info_stable_1.json', 'w') as f:
    json.dump(new_data, f, indent=4)

print(f"Successfully extracted {total_apis} APIs with stability '1.0.0' into rust_api_info_stable_1.json")