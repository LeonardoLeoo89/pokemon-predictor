



import json
from typing import Generator, Dict, Any, List, Optional
import pandas as pd


def load_jsonl(filepath: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:










    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))
    return records


def stream_jsonl(filepath: str) -> Generator[Dict[str, Any], None, None]:










    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                yield json.loads(line_str)


def parse_metadata_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:










    data = []
    for battle in records:
        row = {
            "battle_id": battle.get("battle_id"),
        }
        if "player_won" in battle:
            row["player_won"] = int(battle["player_won"])
        data.append(row)

    return pd.DataFrame(data)


def load_type_chart(filepath: str) -> Dict[str, Any]:









    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)