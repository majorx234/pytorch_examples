import logging
from typing import Dict, List, Set, Tuple

from torch.utils.data import Dataset


class LanguageDataset(Dataset):
    def __init__(self, rows) -> None:
        self.rows: List[Dict[str, str]] = [row["row"] for row in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx) -> Tuple[str, str]:
        row = self.rows[idx]
        return row["labels"], row["text"]
