"""Layer 12 — approval CLI.

The one manual step. Proposals land status='proposed'; a human approves in bulk,
never per-item (per-item approval is the fatigue trap machine-derived metadata
exists to avoid). Approval is the only place a pack changes status.

    python -m pipelines.catalog.approve list
    python -m pipelines.catalog.approve approve --batch --all
    python -m pipelines.catalog.approve approve --batch <pack_id> <pack_id> ...
    python -m pipelines.catalog.approve reject <pack_id> ...
"""

from __future__ import annotations

import argparse
import sqlite3

from ledger import connect, migrate
from pipelines.catalog import config


def list_proposed(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT p.id, p.title, p.rationale, COUNT(m.product_id) AS n
        FROM packs p LEFT JOIN pack_members m ON m.pack_id = p.id
        WHERE p.status = 'proposed'
        GROUP BY p.id ORDER BY p.created_at
        """
    ).fetchall()


def set_status(conn: sqlite3.Connection, status: str, pack_ids: list[str]) -> int:
    cur = conn.executemany(
        "UPDATE packs SET status = ? WHERE id = ? AND status = 'proposed'",
        [(status, pid) for pid in pack_ids],
    )
    return cur.rowcount


def approve_all(conn: sqlite3.Connection) -> int:
    cur = conn.execute("UPDATE packs SET status = 'approved' WHERE status = 'proposed'")
    return cur.rowcount


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="approve", description="Approve proposed packs in bulk.")
    parser.add_argument("--db", default=str(config.DB_PATH))
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="show proposed packs")

    ap = sub.add_parser("approve", help="approve proposed packs")
    ap.add_argument("--batch", action="store_true", required=True, help="bulk approval (required)")
    ap.add_argument("--all", action="store_true", help="approve every proposed pack")
    ap.add_argument("pack_ids", nargs="*")

    rj = sub.add_parser("reject", help="reject proposed packs")
    rj.add_argument("pack_ids", nargs="+")

    args = parser.parse_args(argv)
    migrate(args.db)
    conn = connect(args.db)
    try:
        if args.cmd == "list":
            rows = list_proposed(conn)
            if not rows:
                print("no proposed packs")
            for r in rows:
                print(f"{r['id']}  [{r['n']} items]  {r['title']}")
                if r["rationale"]:
                    print(f"    {r['rationale']}")
            return 0

        if args.cmd == "approve":
            n = approve_all(conn) if args.all else set_status(conn, "approved", args.pack_ids)
            print(f"approved {n} pack(s)")
            return 0

        if args.cmd == "reject":
            n = set_status(conn, "rejected", args.pack_ids)
            print(f"rejected {n} pack(s)")
            return 0
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
