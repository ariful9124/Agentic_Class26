import sqlite3
import re
import random
from collections import Counter

def extract_sql(raw: str) -> str:
    """
    Extract SQL from model output. The base model often appends Answer:, Explanation:,
    or hallucinated query results. Take only the SQL part (before those markers).
    """
    if not raw:
        return ""
    for sep in ["\nAnswer:", "\nExplanation:", "\nSQL:"]:
        idx = raw.find(sep)
        if idx >= 0:
            raw = raw[:idx]
    return raw.strip()


def normalize_sql(sql: str) -> str:
    """
    Normalize SQL for comparison:
    - Strip <|end_of_text|> and similar special tokens
    - Treat single and double quotes as equivalent
    - Normalize whitespace
    """
    if not sql:
        return ""
    sql = re.sub(r"<\|[^|]+\|>", "", sql)
    sql = sql.strip()
    sql = sql.replace('"', "'")
    sql = " ".join(sql.split())
    sql = sql.rstrip(";")
    return sql


def _extract_literals(sql: str) -> tuple[list[str], list[float]]:
    """Extract string and numeric literals from a SQL query."""
    strings = re.findall(r"'([^']*)'", sql) + re.findall(r'"([^"]*)"', sql)
    numbers = [float(m.group(1)) for m in re.finditer(
        r"(?<![a-zA-Z_])(\d+(?:\.\d+)?)(?![a-zA-Z_])", sql
    ) if m.group(1)]
    return strings, numbers


def _build_db(schema: str, str_lits: list[str], num_lits: list[float],
              seed: int = 0, n_rows: int = 50) -> sqlite3.Connection:
    """Build an in-memory SQLite DB from schema, seeded with query literals."""
    rng = random.Random(seed)
    # Make text columns case-insensitive
    nocase = re.sub(
        r"\b(\w+(?:\(\d+\))?)\s*(?=,|\))",
        lambda m: m.group(0) + " COLLATE NOCASE"
        if any(k in m.group(1).upper() for k in ("VARCHAR", "TEXT", "CHAR"))
        else m.group(0),
        schema,
    )
    conn = sqlite3.connect(":memory:")
    for stmt in [s.strip() for s in nocase.split(";") if s.strip()]:
        if not stmt.upper().startswith("CREATE"):
            continue
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError:
            continue
        hdr = re.search(r"CREATE\s+TABLE\s+(\w+)\s*\((.+)\)", stmt,
                        re.IGNORECASE | re.DOTALL)
        if not hdr:
            continue
        table, cols = hdr.group(1), []
        for part in hdr.group(2).split(","):
            toks = part.strip().split()
            if len(toks) >= 2 and not toks[0].upper().startswith(
                ("PRIMARY", "FOREIGN", "UNIQUE", "CHECK", "CONSTRAINT")
            ):
                is_num = any(k in toks[1].upper() for k in
                             ("INT", "REAL", "FLOAT", "DOUBLE", "NUMERIC"))
                pool = (list(set(num_lits)) + [float(i * 7 + len(cols) * 3 + 1) for i in range(8)]
                        if is_num else
                        list(set(str_lits)) + [f"{toks[0]}_v{i}" for i in range(6)])
                rng.shuffle(pool)
                cols.append(pool)
        for i in range(n_rows):
            vals = [p[i % len(p)] for p in cols]
            try:
                conn.execute(f"INSERT INTO {table} VALUES ({','.join('?' * len(vals))})", vals)
            except sqlite3.Error:
                break
    conn.commit()
    return conn


def sql_matches(generated: str, expected, schema: str = "") -> bool:
    """
    Check if generated SQL matches expected.
    With schema: execution-based (runs both on multiple seeded SQLite DBs).
    Without schema: normalized string comparison.
    """
    gen_sql = normalize_sql(extract_sql(generated))
    if not gen_sql:
        return False
    expected_list = [expected] if isinstance(expected, str) else expected

    if not schema:
        return any(gen_sql == normalize_sql(e) for e in expected_list)

    # Gather all literals from both sides for realistic seed data
    all_strs, all_nums = [], []
    for sql in [gen_sql] + [normalize_sql(e) for e in expected_list]:
        s, n = _extract_literals(sql)
        all_strs.extend(s)
        all_nums.extend(n)

    n_dbs = 5
    for exp in expected_list:
        exp_sql = normalize_sql(exp)
        if all(_exec_match(gen_sql, exp_sql, schema, all_strs, all_nums, seed=i * 97 + 13)
               for i in range(n_dbs)):
            return True
    return False


def _exec_match(gen_sql, exp_sql, schema, strs, nums, seed) -> bool:
    """Run both queries on one seeded DB and compare result multisets."""
    conn = _build_db(schema, strs, nums, seed=seed)
    try:
        gr = conn.execute(gen_sql).fetchall()
        er = conn.execute(exp_sql).fetchall()
    except sqlite3.Error:
        return False
    finally:
        conn.close()
    if not gr and not er:
        return True
    if gr and er and len(gr[0]) != len(er[0]):
        return False
    return Counter(gr) == Counter(er)