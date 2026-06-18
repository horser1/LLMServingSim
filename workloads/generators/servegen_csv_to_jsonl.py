"""Convert ServeGen CSV workloads to LLMServingSim JSONL format.

Usage:
    python -m workloads.generators.servegen_csv_to_jsonl <input.csv> [output.jsonl]

If output path is omitted, replaces .csv with .jsonl in the same directory.
"""

import csv
import json
import sys
from pathlib import Path


def convert(csv_path: str, jsonl_path: str | None = None) -> str:
    csv_path = Path(csv_path)
    if jsonl_path is None:
        jsonl_path = csv_path.with_suffix('.jsonl')
    else:
        jsonl_path = Path(jsonl_path)

    count = 0
    with open(csv_path, newline='') as fin, open(jsonl_path, 'w') as fout:
        reader = csv.DictReader(fin)
        for row in reader:
            record = {
                'input_toks': int(row['input_tokens']),
                'output_toks': int(row['output_tokens']),
                'arrival_time_ns': int(round(float(row['timestamp']) * 1_000_000_000)),
            }
            fout.write(json.dumps(record) + '\n')
            count += 1

    print(f'Converted {count} requests: {csv_path} -> {jsonl_path}')
    return str(jsonl_path)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__.strip())
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
