import json
import pandas as pd

path = "test.jsonl"
df = pd.read_json(path, lines=True)
# Reformatting it for IR
out_dir = "reformatted_data.jsonl"
with open(out_dir, "w") as f:
    f.write("")
for idx, row in df.iterrows():
    new_entry = ({
        "id": row["qid"],
        "contents": {
            "problem_description": row["context"],
            "diagnosis": row["diagnosis"]
        }
            })
    with open(out_dir, "a") as f:
        f.write(json.dumps(new_entry) + "\n")
