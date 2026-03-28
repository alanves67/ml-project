"""
Модуль для работы с PostgreSQL базой данных
Использует psycopg2-binary (без компиляции)
"""

import os
from datetime import datetime
from typing import List, Dict, Any
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class PredictionDatabase:
    """
    Класс для работы с базой данных предсказаний
    """

    def __init__(self, database_url: str = None):
        if database_url is None:
            database_url = os.getenv(
                'DATABASE_URL',
                'postgresql://ml_user:ml_pass@localhost:5432/ml_db'
            )

        self.database_url = database_url
        self.pool = None
        self.connect()

    def connect(self):
        """Создает пул соединений с базой данных"""
        try:
            self.pool = SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                dsn=self.database_url
            )
            logger.info(f"Connected to database")

            # Создаем таблицу, если не существует
            self._create_table()

        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise

    def _create_table(self):
        """Создает таблицу для предсказаний"""
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                            CREATE TABLE IF NOT EXISTS predictions
                            (
                                id
                                SERIAL
                                PRIMARY
                                KEY,
                                timestamp
                                TIMESTAMP
                                DEFAULT
                                CURRENT_TIMESTAMP,
                                features
                                TEXT
                                NOT
                                NULL,
                                prediction
                                FLOAT
                                NOT
                                NULL,
                                user_ip
                                VARCHAR
                            (
                                45
                            ),
                                request_id VARCHAR
                            (
                                36
                            )
                                )
                            """)
                conn.commit()
            logger.info("Table 'predictions' ready")
        finally:
            self.pool.putconn(conn)

    def log_prediction(self, features: Dict[str, float], prediction: float,
                       user_ip: str = None, request_id: str = None) -> int:
        """Сохраняет предсказание в базу"""
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                            INSERT INTO predictions (features, prediction, user_ip, request_id)
                            VALUES (%s, %s, %s, %s) RETURNING id
                            """, (json.dumps(features), prediction, user_ip, request_id))
                conn.commit()
                record_id = cur.fetchone()[0]
                logger.debug(f"Prediction logged with id: {record_id}")
                return record_id
        finally:
            self.pool.putconn(conn)

    def get_recent_predictions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Получает последние предсказания"""
        conn = self.pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                            SELECT id, timestamp, features, prediction, user_ip, request_id
                            FROM predictions
                            ORDER BY id DESC
                                LIMIT %s
                            """, (limit,))
                records = cur.fetchall()

                result = []
                for record in records:
                    record['features'] = json.loads(record['features'])
                    record['timestamp'] = record['timestamp'].isoformat()
                    result.append(dict(record))
                return result
        finally:
            self.pool.putconn(conn)

    def get_prediction_stats(self) -> Dict[str, Any]:
        """Получает статистику по предсказаниям"""
        conn = self.pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                            SELECT COUNT(*)        as total_predictions,
                                   AVG(prediction) as avg_prediction,
                                   MIN(prediction) as min_prediction,
                                   MAX(prediction) as max_prediction
                            FROM predictions
                            """)
                stats = cur.fetchone()
                return {
                    'total_predictions': stats['total_predictions'] or 0,
                    'avg_prediction': float(stats['avg_prediction']) if stats['avg_prediction'] else None,
                    'min_prediction': float(stats['min_prediction']) if stats['min_prediction'] else None,
                    'max_prediction': float(stats['max_prediction']) if stats['max_prediction'] else None
                }
        finally:
            self.pool.putconn(conn)

    def health_check(self) -> bool:
        """Проверяет доступность базы"""
        try:
            conn = self.pool.getconn()
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
            self.pool.putconn(conn)
            return result is not None
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False