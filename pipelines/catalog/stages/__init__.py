"""Pipeline stages. Each stage is a plain function over explicit inputs that
returns a result dataclass; persistence to the ledger is a separate concern
(stages/write.py). This keeps every stage testable without a database."""
