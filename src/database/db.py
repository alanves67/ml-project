"""
Модуль для работы с SQLite базой данных (легкая версия)
"""

import sqlite3
import json
from typing import List, Dict, Any
import os

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class PredictionDatabase:
    """Класс для работы с SQLite базой данных"""

    def __init__(self, database_url: str = None):
        # SQLite файл будет в папке проекта
        self.db_path = 'predictions.db'
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_table()
        logger.info(f"Connected to SQLite database: {self.db_path}")

    def _create_table(self):
        """Создает таблицу для предсказаний"""
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                features TEXT NOT NULL,
                prediction REAL NOT NULL,
                user_ip TEXT,
                request_id TEXT
            )
        ''')
        self.conn.commit()
        logger.info("Table 'predictions' ready")

    def log_prediction(self, features: Dict[str, float], prediction: float,
                       user_ip: str = None, request_id: str = None) -> int:
        """Сохраняет предсказание в базу"""
        cursor = self.conn.execute(
            "INSERT INTO predictions (features, prediction, user_ip, request_id) VALUES (?, ?, ?, ?)",
            (json.dumps(features), prediction, user_ip, request_id)
        )
        self.conn.commit()
        logger.debug(f"Prediction logged with id: {cursor.lastrowid}")
        return cursor.lastrowid

    def get_recent_predictions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Получает последние предсказания"""
        cursor = self.conn.execute(
            "SELECT id, timestamp, features, prediction, user_ip, request_id FROM predictions ORDER BY id DESC LIMIT ?",
            (limit,)
        )
        records = cursor.fetchall()

        result = []
        for record in records:
            result.append({
                'id': record[0],
                'timestamp': record[1],
                'features': json.loads(record[2]),
                'prediction': record[3],
                'user_ip': record[4],
                'request_id': record[5]
            })
        return result

    def get_prediction_stats(self) -> Dict[str, Any]:
        """Получает статистику по предсказаниям"""
        cursor = self.conn.execute("""
            SELECT 
                COUNT(*) as total,
                AVG(prediction) as avg,
                MIN(prediction) as min,
                MAX(prediction) as max
            FROM predictions
        """)
        row = cursor.fetchone()
        return {
            'total_predictions': row[0] or 0,
            'avg_prediction': row[1] if row[1] is not None else None,
            'min_prediction': row[2] if row[2] is not None else None,
            'max_prediction': row[3] if row[3] is not None else None
        }

    def health_check(self) -> bool:
        """Проверяет доступность базы"""
        try:
            self.conn.execute("SELECT 1")
            return True
        except Exception:
            return False