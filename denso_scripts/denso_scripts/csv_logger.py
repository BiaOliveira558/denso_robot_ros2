#!/usr/bin/env python3
import csv
import os
from datetime import datetime


class CsvLogger:
    def __init__(self, output_dir: str, prefix: str, columns: list[str]):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{prefix}_{timestamp}.csv'
        self.filepath = os.path.join(output_dir, filename)

        self.columns = columns
        self._file = open(self.filepath, mode='w', newline='')
        self._writer = csv.DictWriter(self._file, fieldnames=self.columns)
        self._writer.writeheader()
        self._file.flush()

    def log(self, row: dict):
        self._writer.writerow({col: row.get(col, '') for col in self.columns})
        self._file.flush()

    def close(self):
        self._file.close()