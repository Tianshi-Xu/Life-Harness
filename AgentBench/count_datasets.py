import json
import glob

# ALFWorld
with open('data/alfworld/new_std.json') as f:
    d = json.load(f)
    print("ALFWorld std:", sum(len(v) for v in d.values()))
with open('data/alfworld/train_valid.json') as f:
    d = json.load(f)
    print("ALFWorld env_train:", sum(len(v) for v in d.values()))

# DBBench
with open('data/dbbench/standard.jsonl') as f:
    print("DBBench std:", sum(1 for _ in f if _.strip()))
with open('data/dbbench/db_out_new.jsonl') as f:
    print("DBBench env_train:", sum(1 for _ in f if _.strip()))

# KG
with open('data/knowledgegraph/std.json') as f:
    print("KG std:", len(json.load(f)))
with open('data/knowledgegraph/kg_rl_all.json') as f:
    print("KG env_train:", len(json.load(f)))

# OS Interaction
os_std_files = glob.glob('data/os_interaction/data/*/*.json')
# Need to count items in each file if they are lists, or if each file is an item
os_std_count = 0
for file in os_std_files:
    with open(file) as f:
        try:
            d = json.load(f)
            if isinstance(d, list): os_std_count += len(d)
            else: os_std_count += 1
        except: pass
print("OS std:", os_std_count)

with open('data/os_interaction/train_0317/training.json') as f:
    d = json.load(f)
    if isinstance(d, list): print("OS env_train:", len(d))
    else: print("OS env_train: 1")

