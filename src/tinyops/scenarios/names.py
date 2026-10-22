"""Word pools. Train/val/test use TRAIN_POOLS; test_ood uses OOD_POOLS (no overlap, see tests)."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class NamePools:
    name: str
    namespaces: tuple[str, ...]
    apps: tuple[str, ...]
    registry: str
    env_vars: tuple[str, ...]
    int_config_keys: tuple[str, ...]
    url_config_keys: tuple[str, ...]
    nodes: tuple[str, ...]
    ports: tuple[int, ...]
    cron_jobs: tuple[str, ...]
    downstream: tuple[str, ...]  # services named in benign/misleading log lines


TRAIN_POOLS = NamePools(
    name="train",
    namespaces=(
        "payments", "checkout", "orders", "inventory", "auth", "search", "catalog", "billing",
        "notifications", "shipping", "returns", "recommendations", "reviews", "loyalty",
        "pricing", "fulfillment", "analytics", "messaging", "media", "identity", "support",
        "wishlist",
    ),
    apps=(
        "checkout-api", "payment-worker", "cart-service", "order-processor", "inventory-sync",
        "auth-gateway", "search-indexer", "catalog-api", "billing-engine", "notify-dispatcher",
        "shipping-tracker", "returns-portal", "reco-engine", "review-service", "loyalty-api",
        "pricing-service", "fulfillment-worker", "analytics-ingest", "chat-relay",
        "media-resizer", "identity-provider", "support-desk", "wishlist-api", "session-store",
        "email-sender", "sms-gateway", "tax-calculator", "coupon-service", "address-validator",
        "fraud-scorer", "rate-limiter", "invoice-renderer",
    ),
    registry="registry.acme.io",
    env_vars=(
        "DATABASE_URL", "REDIS_URL", "API_KEY", "SECRET_KEY", "JWT_SECRET", "AMQP_URL",
        "SMTP_HOST", "STRIPE_API_KEY", "SESSION_SECRET", "ELASTIC_URL", "CACHE_HOST",
        "PAYMENT_GATEWAY_URL",
    ),
    int_config_keys=(
        "MAX_CONN", "WORKER_COUNT", "POOL_SIZE", "RETRY_LIMIT", "BATCH_SIZE", "QUEUE_DEPTH",
        "TIMEOUT_SECONDS", "CACHE_TTL",
    ),
    url_config_keys=("UPSTREAM_URL", "CALLBACK_URL", "WEBHOOK_URL", "BACKEND_URL"),
    nodes=("node-a1", "node-a2", "node-a3", "node-b1", "node-b2", "node-b3", "node-c1"),
    ports=(3000, 5000, 8000, 8081, 9000),
    cron_jobs=("nightly-cleanup", "report-export", "cache-warmup", "stale-session-sweeper"),
    downstream=("payments-gateway", "user-directory", "pricing-cache", "mail-relay"),
)

OOD_POOLS = NamePools(
    name="ood",
    namespaces=(
        "telemetry", "ledger", "render-farm", "kyc", "video-transcode", "genomics", "ticketing",
        "scheduler", "geo-routing", "ml-serving", "archive",
    ),
    apps=(
        "span-collector", "ledger-writer", "frame-renderer", "kyc-verifier", "transcode-worker",
        "variant-caller", "ticket-router", "cron-orchestrator", "route-planner", "model-server",
        "cold-archiver", "metrics-rollup", "feature-store", "audit-sink",
    ),
    registry="ghcr.io/initech",
    env_vars=(
        "FEATURE_STORE_ADDR", "KAFKA_BOOTSTRAP", "OTEL_ENDPOINT", "S3_BUCKET_NAME", "VAULT_ADDR",
        "GRPC_TARGET",
    ),
    int_config_keys=("SHARD_COUNT", "FETCH_PARALLELISM", "MAX_INFLIGHT", "LEASE_SECONDS"),
    url_config_keys=("INGEST_ENDPOINT", "REGISTRY_MIRROR_URL"),
    nodes=("worker-eu-1", "worker-eu-2", "worker-us-1", "worker-us-2"),
    ports=(4000, 7000, 8800),
    cron_jobs=("compaction-run", "snapshot-prune"),
    downstream=("schema-registry", "blob-gateway", "quota-service"),
)

BAD_INT_VALUES = ("ten", "many", "auto", "lots", "unlimited", "twenty")
BAD_URL_VALUES = ("htp://backend.internal", "backend.internal:://", "http//backend.internal",
                  "://missing-scheme", "http://")


def pods_hash(rng: random.Random) -> str:
    return "".join(rng.choice("0123456789abcdef") for _ in range(6))


def pod_suffix(rng: random.Random) -> str:
    return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(5))


def semver(rng: random.Random) -> str:
    return f"{rng.randint(1, 3)}.{rng.randint(0, 9)}.{rng.randint(1, 9)}"
