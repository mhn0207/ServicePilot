"""持久化合成账单和退款核验单。沙盒没有支付渠道连接。"""

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


DEMO_USER_ID = "u1001"
DEMO_ORDER_ID = "ORD-20261024"


class SandboxStore:
    def __init__(self, path: Path):
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=3)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db:
            db.execute("PRAGMA journal_mode = WAL").fetchone()
            db.executescript("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    product TEXT NOT NULL,
                    currency TEXT NOT NULL,
                    total_cents INTEGER NOT NULL CHECK (total_cents > 0)
                );
                CREATE TABLE IF NOT EXISTS billing_ledger (
                    entry_id TEXT PRIMARY KEY,
                    order_id TEXT NOT NULL REFERENCES orders(order_id),
                    status TEXT NOT NULL CHECK (status IN ('settled', 'authorization_hold')),
                    amount_cents INTEGER NOT NULL CHECK (amount_cents > 0)
                );
                CREATE TABLE IF NOT EXISTS refund_reviews (
                    review_id TEXT PRIMARY KEY,
                    order_id TEXT NOT NULL UNIQUE REFERENCES orders(order_id),
                    status TEXT NOT NULL CHECK (status = 'pending_manual_review'),
                    created_at TEXT NOT NULL
                );
            """)
            with db:
                db.execute("""
                    INSERT INTO orders VALUES (?, ?, 'ServicePilot Pro 月度订阅', 'CNY', 19900)
                    ON CONFLICT(order_id) DO NOTHING
                """, (DEMO_ORDER_ID, DEMO_USER_ID))
                db.executemany("""
                    INSERT INTO billing_ledger VALUES (?, ?, ?, 19900)
                    ON CONFLICT(entry_id) DO NOTHING
                """, [("CHG-DEMO-01", DEMO_ORDER_ID, "settled"),
                      ("CHG-DEMO-02", DEMO_ORDER_ID, "authorization_hold")])

    def order_ledger(self, user_id: str, order_id: str) -> dict | None:
        with closing(self._connect()) as db:
            order = db.execute("""
                SELECT order_id, product, currency, total_cents FROM orders
                WHERE user_id = ? AND order_id = ?
            """, (user_id, order_id)).fetchone()
            if order is None:
                return None
            entries = db.execute("""
                SELECT entry_id, status, amount_cents FROM billing_ledger
                WHERE order_id = ? ORDER BY entry_id
            """, (order_id,)).fetchall()
        return {**dict(order), "entries": [dict(entry) for entry in entries]}

    def submit_refund_review(self, user_id: str, order_id: str) -> dict | None:
        """当前沙盒每个订单对应一张核验单，跨会话与进程重试复用原单。"""
        with closing(self._connect()) as db, db:
            db.execute("""
                INSERT INTO refund_reviews (review_id, order_id, status, created_at)
                SELECT ?, order_id, 'pending_manual_review', ? FROM orders
                WHERE user_id = ? AND order_id = ?
                ON CONFLICT(order_id) DO NOTHING
            """, (f"RFD-{uuid4().hex.upper()}", datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  user_id, order_id))
            return self._find_review(db, user_id, order_id=order_id)

    def refund_review(self, user_id: str, review_id: str) -> dict | None:
        with closing(self._connect()) as db:
            return self._find_review(db, user_id, review_id=review_id)

    @staticmethod
    def _find_review(db, user_id, *, order_id=None, review_id=None) -> dict | None:
        row = db.execute("""
            SELECT r.review_id, r.order_id, r.status, r.created_at FROM refund_reviews r
            JOIN orders o ON o.order_id = r.order_id
            WHERE o.user_id = ? AND (r.order_id = ? OR r.review_id = ?)
        """, (user_id, order_id, review_id)).fetchone()
        return dict(row) if row is not None else None
