"""Central configuration: environment variables, vocabularies, observation window.

All settings come from environment variables (optionally loaded from a git-ignored
.env file). Nothing secret is hardcoded.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# abspath (not resolve): resolve() follows Windows junction/redirected folders to a path that may differ
PROJECT_ROOT = Path(os.path.abspath(__file__)).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# --------------------------------------------------------------------------- #
# Vocabularies. These mirror the CHECK constraints in sql/04_create_constraints.sql
# (the loader fails loudly if the two ever drift apart).
# --------------------------------------------------------------------------- #
ACQUISITION_CHANNELS = ["organic", "paid_search", "social", "email", "referral", "affiliate"]
CUSTOMER_SEGMENTS = ["budget", "standard", "premium", "business"]
ORDER_STATUSES = ["completed", "cancelled", "refunded"]
PAYMENT_METHODS = ["card", "paypal", "bank_transfer", "gift_card", "wallet"]
EVENT_TYPES = ["page_view", "product_view", "add_to_cart", "checkout_start", "purchase"]
DEVICE_TYPES = ["desktop", "mobile", "tablet"]
REFERRER_TYPES = ["direct", "organic_search", "paid_search", "social", "email", "referral"]

# --------------------------------------------------------------------------- #
# Temporal contract: ALL timestamps are naive UTC. The observation window is
# half-open: [OBS_START, OBS_END). Signups happen in the first ~2 years so that
# every customer has at least 6 months of observable behaviour.
# --------------------------------------------------------------------------- #
OBS_START = datetime(2023, 1, 1)
SIGNUP_END = datetime(2025, 1, 1)
OBS_END = datetime(2025, 7, 1)


# --------------------------------------------------------------------------- #
# ML experiment design. Partitions are defined by feature CUTOFF dates (snapshot_ts). Each snapshot's
# label looks 30 days ahead, so the first cutoff of a partition must be >= (last cutoff of the previous
# partition + 30 days): no outcome window of an earlier partition reaches into a later one.
# --------------------------------------------------------------------------- #
OUTCOME_DAYS = 30
SPLITS = {
    "train": ("2023-04-01", "2024-06-01"),   # 15 monthly snapshots
    "validation": ("2024-08-01", "2024-11-01"),   # 4 snapshots (July 2024 is the purge gap)
    "test": ("2025-01-01", "2025-06-01"),    # 6 snapshots (December 2024 is the purge gap); touched once
}
SCORING_SNAPSHOT = "2025-07-01"              # features only, no complete outcome window yet

# Illustrative, configurable business assumptions for threshold selection (NOT measured; synthetic data).
BUSINESS = {
    "contact_cost": 0.50,          # $ per customer contacted (message / handling)
    "save_rate": 0.25,             # P(a would-be churner stays after the offer)
    "value_if_saved": 20.0,        # $ margin kept in the next 30 days when a churner is saved
    "redeem_rate": 0.50,           # P(a non-churner redeems the offer anyway)
    "discount_cost": 4.0,          # $ discount given away on redemption
}


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


@dataclass(frozen=True)
class Settings:
    # connection
    host: str
    port: str
    database: str
    username: str
    password: str
    driver: str
    encrypt: str
    trust_certificate: str
    login_timeout: int
    query_timeout: int
    pool_size: int
    # synthetic data
    random_seed: int
    num_users: int
    num_orders: int
    num_events: int
    batch_size: int
    feature_cutoff_date: str
    data_dir: Path

    @property
    def uses_windows_auth(self) -> bool:
        return not self.username

    @staticmethod
    def from_env() -> "Settings":
        return Settings(
            host=_env("SQL_SERVER_HOST", "localhost"),
            port=_env("SQL_SERVER_PORT"),
            database=_env("SQL_SERVER_DATABASE", "CustomerFeatureStore"),
            username=_env("SQL_SERVER_USERNAME"),
            password=os.getenv("SQL_SERVER_PASSWORD", ""),
            driver=_env("SQL_SERVER_DRIVER", "ODBC Driver 18 for SQL Server"),
            # Driver 18 encrypts by default; we default to yes and never silently weaken it.
            encrypt=_env("SQL_SERVER_ENCRYPT", "yes"),
            trust_certificate=_env("SQL_SERVER_TRUST_CERTIFICATE", "no"),
            login_timeout=int(_env("SQL_SERVER_LOGIN_TIMEOUT", "15")),
            query_timeout=int(_env("SQL_SERVER_QUERY_TIMEOUT", "0")),  # 0 = no limit
            pool_size=int(_env("SQL_SERVER_POOL_SIZE", "5")),
            random_seed=int(_env("RANDOM_SEED", "42")),
            num_users=int(_env("NUM_USERS", "50000")),
            num_orders=int(_env("NUM_ORDERS", "200000")),
            num_events=int(_env("NUM_EVENTS", "1000000")),
            batch_size=int(_env("BATCH_SIZE", "10000")),
            feature_cutoff_date=_env("FEATURE_CUTOFF_DATE", "2025-06-30"),
            data_dir=Path(_env("DATA_DIR", str(PROJECT_ROOT / "data"))),
        )


def get_settings() -> Settings:
    return Settings.from_env()


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
