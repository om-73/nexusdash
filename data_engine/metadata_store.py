import sqlite3
import json
import hashlib
from datetime import datetime
import os

DB_FILE = "metadata.db"

class MetadataStore:
    def __init__(self):
        self.conn = None
        self.init_db()

    def get_conn(self):
        # Check if we need to reconnect (e.g. valid thread)
        # For simple usage, creating a new connection per request is safer for threads
        return sqlite3.connect(DB_FILE, check_same_thread=False)

    def init_db(self):
        conn = self.get_conn()
        cur = conn.cursor()
        
        # 1. Datasets Table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY,
                name TEXT,
                source_type TEXT,
                created_at TIMESTAMP
            )
        """)
        
        # 2. Runs Table (History)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id TEXT,
                row_count INTEGER,
                col_count INTEGER,
                schema_hash TEXT,
                columns_json TEXT, 
                timestamp TIMESTAMP,
                FOREIGN KEY(dataset_id) REFERENCES datasets(id)
            )
        """)
        
        # 3. Contracts Table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS contracts (
                dataset_id TEXT PRIMARY KEY,
                contract_json TEXT,
                updated_at TIMESTAMP,
                FOREIGN KEY(dataset_id) REFERENCES datasets(id)
            )
        """)
        
        # 4. Feature Store Tables
        cur.execute("""
            CREATE TABLE IF NOT EXISTS features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                description TEXT,
                created_at TIMESTAMP
            )
        """)
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS feature_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature_id INTEGER,
                version TEXT,
                logic_code TEXT,
                created_at TIMESTAMP,
                FOREIGN KEY(feature_id) REFERENCES features(id)
            )
        """)
        
        conn.commit()
        conn.close()

    def register_dataset(self, dataset_id, name, source_type):
        conn = self.get_conn()
        try:
            cur = conn.cursor()
            cur.execute("INSERT OR IGNORE INTO datasets (id, name, source_type, created_at) VALUES (?, ?, ?, ?)",
                        (dataset_id, name, source_type, datetime.now()))
            conn.commit()
        finally:
            conn.close()

    def log_run(self, dataset_id, df, snapshot_path=None):
        conn = self.get_conn()
        try:
            row_count = len(df)
            col_count = len(df.columns)
            columns = df.columns.tolist()
            # Simple schema hash: sorted column names + types
            # Using simple str representation for robustness
            schema_str = str(sorted([(c, str(t)) for c, t in df.dtypes.items()]))
            schema_hash = hashlib.md5(schema_str.encode()).hexdigest()
            
            cur = conn.cursor()
            
            # Simple migration for existing dev DB: add column if not exists
            try:
                cur.execute("ALTER TABLE runs ADD COLUMN snapshot_path TEXT")
            except sqlite3.OperationalError:
                pass # Column likely exists or other issue we can ignore for now

            cur.execute("""
                INSERT INTO runs (dataset_id, row_count, col_count, schema_hash, columns_json, timestamp, snapshot_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (dataset_id, row_count, col_count, schema_hash, json.dumps(columns), datetime.now(), snapshot_path))
            conn.commit()
        finally:
            conn.close()

    def get_run_history(self, dataset_id, limit=10):
        conn = self.get_conn()
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT row_count, col_count, schema_hash, timestamp, snapshot_path
                FROM runs 
                WHERE dataset_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (dataset_id, limit))
            return cur.fetchall()
        finally:
            conn.close()

    def save_contract(self, dataset_id, contract_dict):
        conn = self.get_conn()
        try:
            cur = conn.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO contracts (dataset_id, contract_json, updated_at)
                VALUES (?, ?, ?)
            """, (dataset_id, json.dumps(contract_dict), datetime.now()))
            conn.commit()
        finally:
            conn.close()

    def get_contract(self, dataset_id):
        conn = self.get_conn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT contract_json FROM contracts WHERE dataset_id = ?", (dataset_id,))
            row = cur.fetchone()
            return json.loads(row[0]) if row else None
        finally:
            conn.close()

    # --- Feature Store Methods ---
    def register_feature(self, name, description, version, logic_code):
        conn = self.get_conn()
        try:
            cur = conn.cursor()
            # 1. Get or Create Feature ID
            cur.execute("SELECT id FROM features WHERE name = ?", (name,))
            row = cur.fetchone()
            if row:
                feature_id = row[0]
            else:
                cur.execute("INSERT INTO features (name, description, created_at) VALUES (?, ?, ?)", 
                            (name, description, datetime.now()))
                feature_id = cur.lastrowid
            
            # 2. Add Version
            cur.execute("""
                INSERT INTO feature_versions (feature_id, version, logic_code, created_at)
                VALUES (?, ?, ?, ?)
            """, (feature_id, version, logic_code, datetime.now()))
            conn.commit()
            return feature_id
        finally:
            conn.close()

        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT f.name, f.description, v.version, v.logic_code 
                FROM features f
                JOIN feature_versions v ON f.id = v.feature_id
                ORDER BY f.name, v.created_at DESC
            """)
            # Post-process to group versions? For now just flat list
            mapped = []
            for r in cur.fetchall():
                mapped.append({"name": r[0], "description": r[1], "version": r[2], "logic": r[3]})
            return mapped
        finally:
            conn.close()

    # --- Dashboard Methods ---
    def save_dashboard(self, dashboard_id, name, layout_json):
        conn = self.get_conn()
        try:
            cur = conn.cursor()
            # Ensure table exists (lazy init for updates)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS dashboards (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    layout_json TEXT,
                    updated_at TIMESTAMP
                )
            """)
            
            cur.execute("""
                INSERT OR REPLACE INTO dashboards (id, name, layout_json, updated_at)
                VALUES (?, ?, ?, ?)
            """, (dashboard_id, name, json.dumps(layout_json), datetime.now()))
            conn.commit()
        finally:
            conn.close()
            
    def get_dashboard(self, dashboard_id):
        conn = self.get_conn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT name, layout_json FROM dashboards WHERE id = ?", (dashboard_id,))
            row = cur.fetchone()
            if row:
                return {"id": dashboard_id, "name": row[0], "layout": json.loads(row[1])}
            return None
        except:
            return None
        finally:
            conn.close()

    def list_dashboards(self):
        conn = self.get_conn()
        try:
            cur = conn.cursor()
            # Check if table exists first
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='dashboards'")
            if not cur.fetchone(): return []
            
            cur.execute("SELECT id, name, updated_at FROM dashboards ORDER BY updated_at DESC")
            return [{"id": r[0], "name": r[1], "updated_at": str(r[2])} for r in cur.fetchall()]
        finally:
            conn.close()
