from __future__ import annotations

import sqlite3
from pathlib import Path


DB_PATH = Path("data/sqlite/ai4h.db")
TABLE_NAME = "ai_augment_feedback"
LIMIT = 20


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    )
    return cur.fetchone() is not None


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB not found: {DB_PATH}")

    with sqlite3.connect(DB_PATH) as conn:
        if not table_exists(conn, TABLE_NAME):
            raise RuntimeError(f"Table not found: {TABLE_NAME}")

        cur = conn.cursor()

        cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
        total_count = int(cur.fetchone()[0])

        cur.execute(f"SELECT COUNT(DISTINCT batch_id) FROM {TABLE_NAME}")
        batch_count = int(cur.fetchone()[0])

        cur.execute(f"SELECT SUM(label), COUNT(*) - SUM(label) FROM {TABLE_NAME}")
        pos_count, neg_count = cur.fetchone()
        pos_count = int(pos_count or 0)
        neg_count = int(neg_count or 0)

        cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE COALESCE(error, 0) = 1")
        error_count = int(cur.fetchone()[0])

        print("db_path:", DB_PATH)
        print("table:", TABLE_NAME)
        print("rows:", total_count)
        print("distinct_batches:", batch_count)
        print("label_1_count:", pos_count)
        print("label_0_count:", neg_count)
        print("error_count:", error_count)
        print()

        print("[batch summary]")
        cur.execute(
            f"""
            SELECT batch_id, query_image_name, COUNT(*) AS n_rows, SUM(label) AS pos_rows, SUM(COALESCE(error, 0)) AS err_rows
            FROM {TABLE_NAME}
            GROUP BY batch_id, query_image_name
            ORDER BY MAX(id) DESC
            LIMIT ?
            """,
            (int(LIMIT),),
        )
        batch_rows = cur.fetchall()
        if not batch_rows:
            print("No rows.")
        else:
            for batch_id, query_image_name, n_rows, pos_rows, err_rows in batch_rows:
                print(
                    f"batch_id={batch_id} rows={int(n_rows)} pos={int(pos_rows or 0)} err={int(err_rows or 0)} "
                    f"query={query_image_name}"
                )

        print()
        print("[latest rows]")
        cur.execute(
            f"""
            SELECT
                id,
                batch_id,
                query_vector_id,
                candidate_vector_id,
                label,
                COALESCE(error, 0),
                query_image_name,
                candidate_image_name
            FROM {TABLE_NAME}
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(LIMIT),),
        )
        latest_rows = cur.fetchall()
        if not latest_rows:
            print("No rows.")
        else:
            for row_id, batch_id, qid, cid, label, error, query_name, candidate_name in latest_rows:
                print(
                    f"id={int(row_id)} batch_id={batch_id} y={int(label)} error={int(error)} "
                    f"qvec={int(qid)} cvec={int(cid)}"
                )
                print(f"  query={query_name}")
                print(f"  candidate={candidate_name}")


if __name__ == "__main__":
    main()
