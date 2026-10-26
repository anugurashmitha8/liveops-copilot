import json
from zetaris_tool import run_sql
import os
from logger import log


def api_items(table):
    row = run_sql(f"SELECT * FROM {table} LIMIT 1", limit=1)[0]
    return json.loads(row["items"])


def live_snapshot():
    assets = api_items("api.assets")
    return {
        "beds": api_items("api.beds"),
        "patients": api_items("api.patients"),
        "cleaning_tasks": api_items("api.cleaning_tasks"),
        "transport_requests": api_items("api.transport_requests"),
        "staff": [item for item in assets if item.get("kind") == "staff"],
        "ambulances": [item for item in assets if item.get("kind") == "ambulance"],
    }


REF_CACHE = os.path.join("data", "reference_cache.json")


def reference_data():
    """Snowflake reference data through Zetaris; falls back to the last saved copy."""
    try:
        ref = {
            "units": run_sql("SELECT * FROM riverside.UNITS", limit=1000),
            "staff_roster": run_sql("SELECT * FROM riverside.STAFF", limit=1000),
            "ambulances": run_sql("SELECT * FROM riverside.AMBULANCES", limit=1000),
            "floors": run_sql("SELECT * FROM riverside.FLOORS", limit=1000),
            "locations": run_sql("SELECT * FROM riverside.LOCATIONS", limit=1000),
        }
    except Exception as err:
        if not os.path.exists(REF_CACHE):
            raise
        log("sources", "reference_cache_used", {"error": str(err)[:300]})
        print("[Sources] Snowflake unavailable: using saved reference data")
        with open(REF_CACHE, encoding="utf-8") as f:
            return json.load(f)
    os.makedirs("data", exist_ok=True)
    with open(REF_CACHE, "w", encoding="utf-8") as f:
        json.dump(ref, f, indent=2, default=str)
    return ref


if __name__ == "__main__":
    live = live_snapshot()
    ref = reference_data()
    print("live_snapshot")
    for name, items in live.items():
        print(f"  {name}: {len(items)}")
    print("reference_data")
    for name, items in ref.items():
        print(f"  {name}: {len(items)}")
