import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from sklearn.model_selection import train_test_split

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE = Path("/scratch/izar/mhamelin/finebio_data/raw_data_v2.jsonl")
OUTPUT_DIR = Path("/scratch/izar/mhamelin/finebio_data")

# Constraints for the Target Subset
# Change TARGET_TOTAL_SAMPLES to 10000 later if you want a bigger set
TARGET_TOTAL_SAMPLES = 10000
MIN_SAMPLES_PER_ACTION = 15
SEED = 42

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def print_stats(data, stage_name="Dataset"):
    """Prints detailed statistics for Actions, Tools, and Targets."""
    print(f"\n{'='*60}")
    print(f"STATS: {stage_name}")
    print(f"{'='*60}")
    print(f"Total Samples: {len(data)}")

    if not data:
        print("No data found.")
        return

    # Aggregate counts
    counters = {
        "Actions": Counter(),
        "Tools": Counter(),
        "Targets": Counter()
    }

    for entry in data:
        counters["Actions"][entry["_meta_action"]] += 1
        counters["Tools"][entry["_meta_tool"]] += 1
        counters["Targets"][entry["_meta_target"]] += 1

    # Print Top 10 for each category
    for category, counter in counters.items():
        print(f"\n--- {category} (Total Unique: {len(counter)}) ---")
        # specific formatting for nice alignment
        sorted_items = counter.most_common(10)
        for item, count in sorted_items:
            pct = (count / len(data)) * 100
            print(f"{count:5d} ({pct:5.1f}%) | {item}")
        
        if len(counter) > 10:
            print(f"... and {len(counter) - 10} more.")

def load_and_clean_data(input_path):
    print(f"Loading {input_path}...")
    valid_data = []
    
    with open(input_path, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                
                # Parse assistant response
                response_str = entry["messages"][1]["content"][0]["text"]
                response = json.loads(response_str)
                
                action = response.get("action", "unknown").lower()
                tool = response.get("tool", "unknown").lower()
                target = response.get("target", "unknown").lower()
                
                # CRITICAL FIX: Only strict-filter the ACTION
                # We allow tool/target to be unknown, otherwise we lose too much data
                invalid_tokens = ["unknown", "nan", "none", "", "n/a"]
                
                if action in invalid_tokens:
                    continue  # We must have a valid action

                # Optional: You can add logic to keep 'unknown' tools if you want
                # or just accept them.
                
                # Store metadata
                entry["_meta_action"] = action
                entry["_meta_tool"] = tool
                entry["_meta_target"] = target
                
                valid_data.append(entry)
                
            except Exception:
                continue
                
    return valid_data

def balance_dataset(data):
    """
    Balances the dataset based on ACTION frequency.
    We assume 'Action' is the primary class we want to equalize.
    """
    # 1. Filter Rare Actions
    action_counts = Counter(d["_meta_action"] for d in data)
    common_actions = {act for act, count in action_counts.items() if count >= MIN_SAMPLES_PER_ACTION}
    
    filtered_data = [d for d in data if d["_meta_action"] in common_actions]
    
    # 2. Cap Frequent Actions
    # Logic: We want roughly uniform distribution across actions
    action_groups = defaultdict(list)
    for d in filtered_data:
        action_groups[d["_meta_action"]].append(d)
        
    final_subset = []
    
    # Calculate dynamic cap
    if not common_actions:
        return []
        
    avg_target = TARGET_TOTAL_SAMPLES // len(common_actions)
    soft_cap = int(avg_target * 3.0) # Allow some imbalance (3x average) to keep data volume
    
    print(f"\n[Balancing Logic] Target: {TARGET_TOTAL_SAMPLES} | Classes: {len(common_actions)} | Avg/Class: {avg_target} | Cap: {soft_cap}")
    
    for action, items in action_groups.items():
        if len(items) > soft_cap:
            # Randomly sample to the cap
            final_subset.extend(random.sample(items, soft_cap))
        else:
            # Keep all if below cap
            final_subset.extend(items)
            
    random.shuffle(final_subset)
    return final_subset

def save_split(data, name):
    output_path = OUTPUT_DIR / f"{name}.jsonl"
    print(f"Saving {len(data)} samples to {output_path}")
    with open(output_path, "w") as f:
        for entry in data:
            # Remove metadata keys (starting with _) before saving to clean up file
            clean_entry = {k: v for k, v in entry.items() if not k.startswith("_meta")}
            f.write(json.dumps(clean_entry) + "\n")

# ==========================================
# MAIN
# ==========================================

def main():
    random.seed(SEED)
    
    # 1. Load & Stats (Raw)
    raw_data = load_and_clean_data(INPUT_FILE)
    print_stats(raw_data, "Raw Valid Data (No Unknowns)")
    
    # 2. Balance & Stats (Filtered)
    dataset = balance_dataset(raw_data)
    print_stats(dataset, f"Balanced Subset (Target ~{TARGET_TOTAL_SAMPLES})")
    
    if not dataset:
        print("Error: Dataset is empty after filtering!")
        return

    # 3. Split (80/10/10)
    # We stratify on ACTION to ensure all verbs are present in all sets
    labels = [d["_meta_action"] for d in dataset]
    
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        dataset, labels, test_size=0.2, stratify=labels, random_state=SEED
    )
    
    val_data, test_data = train_test_split(
        temp_data, test_size=0.5, stratify=temp_labels, random_state=SEED
    )
    
    # 4. Save
    print("\n" + "="*60)
    # save_split(train_data, "train_mini")
    # save_split(val_data, "val_mini")
    # save_split(test_data, "test_mini")
    print("Done!")

if __name__ == "__main__":
    main()
