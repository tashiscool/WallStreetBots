"""Gunicorn configuration for production WallStreetBots deployment.

This configuration is optimized for:
- High availability trading applications
- Prometheus metrics integration
- Structured logging
- Graceful shutdown for active connections
"""

import multiprocessing
import os

# =============================================================================
# Server Socket
# =============================================================================
bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"
backlog = 2048

# =============================================================================
# Worker Processes
# =============================================================================
# Workers = (2 * CPU cores) + 1 for I/O bound applications
# For trading apps, we want more workers to handle concurrent requests
workers = int(os.environ.get("GUNICORN_WORKERS", multiprocessing.cpu_count() * 2 + 1))

# Worker class - sync for Django, can use gevent/eventlet for async
worker_class = os.environ.get("GUNICORN_WORKER_CLASS", "sync")

# Threads per worker (for gthread worker class)
threads = int(os.environ.get("GUNICORN_THREADS", 2))

# Maximum concurrent connections per worker
worker_connections = 1000

# Maximum requests per worker before restart (prevents memory leaks)
max_requests = int(os.environ.get("GUNICORN_MAX_REQUESTS", 10000))
max_requests_jitter = int(os.environ.get("GUNICORN_MAX_REQUESTS_JITTER", 1000))

# =============================================================================
# Timeouts
# =============================================================================
# Worker timeout (seconds) - longer for trading operations
timeout = int(os.environ.get("GUNICORN_TIMEOUT", 120))

# Graceful timeout for finishing requests during shutdown
# Must be >= timeout to avoid killing active trading operations mid-execution
graceful_timeout = int(os.environ.get("GUNICORN_GRACEFUL_TIMEOUT", 120))

# Keep-alive timeout (seconds)
keepalive = int(os.environ.get("GUNICORN_KEEPALIVE", 5))

# =============================================================================
# Server Mechanics
# =============================================================================
# Daemonize - False for Docker
daemon = False

# PID file (useful for process management)
pidfile = "/app/.state/gunicorn.pid"

# User/Group - already set in Dockerfile
# user = "wsb"
# group = "wsb"

# Working directory
chdir = "/app"

# Temp directory for worker heartbeat files
worker_tmp_dir = "/dev/shm"

# =============================================================================
# Logging
# =============================================================================
# Log level
loglevel = os.environ.get("GUNICORN_LOG_LEVEL", "info")

# Access log format (combined with response time)
access_log_format = (
    '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(L)s'
)

# Log to stdout/stderr for Docker
accesslog = "-"
errorlog = "-"

# Capture stdout/stderr from workers
capture_output = True

# Enable JSON logging if structlog is available
if os.environ.get("GUNICORN_JSON_LOGGING", "false").lower() == "true":
    logconfig_dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "json",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "handlers": ["console"],
            "level": "INFO",
        },
    }

# =============================================================================
# Process Naming
# =============================================================================
proc_name = "wallstreetbots"

# =============================================================================
# Server Hooks
# =============================================================================


def on_starting(server):
    """Called just before the master process is initialized."""
    print("[Gunicorn] Starting WallStreetBots server...")


def on_reload(server):
    """Called when workers are reloaded."""
    print("[Gunicorn] Reloading workers...")


def when_ready(server):
    """Called just after the server is started."""
    print(f"[Gunicorn] Server ready. Listening on {bind}")
    print(f"[Gunicorn] Workers: {workers}, Threads: {threads}")


def worker_int(worker):
    """Called when a worker receives SIGINT or SIGQUIT."""
    print(f"[Gunicorn] Worker {worker.pid} interrupted")


def worker_abort(worker):
    """Called when a worker receives SIGABRT."""
    print(f"[Gunicorn] Worker {worker.pid} aborted")


def pre_fork(server, worker):
    """Called just before a worker is forked."""
    pass


def post_fork(server, worker):
    """Called just after a worker has been forked."""
    print(f"[Gunicorn] Worker {worker.pid} spawned")


def post_worker_init(worker):
    """Called just after a worker has initialized the application."""
    # Initialize any per-worker resources here
    pass


def worker_exit(server, worker):
    """Called just after a worker has been exited."""
    print(f"[Gunicorn] Worker {worker.pid} exited")


def nworkers_changed(server, new_value, old_value):
    """Called when the number of workers is changed."""
    print(f"[Gunicorn] Workers changed: {old_value} -> {new_value}")


def on_exit(server):
    """Called just before exiting gunicorn."""
    print("[Gunicorn] Server shutting down...")


# =============================================================================
# SSL (uncomment if using SSL termination at Gunicorn)
# =============================================================================
# keyfile = "/path/to/key.pem"
# certfile = "/path/to/cert.pem"
# ssl_version = "TLSv1_2"
# cert_reqs = 0  # 0 = no client cert, 1 = optional, 2 = required
# ca_certs = None
# suppress_ragged_eofs = True
# do_handshake_on_connect = False

# =============================================================================
# Performance Tuning
# =============================================================================
# Preload application for faster worker spawning (uses more memory)
preload_app = os.environ.get("GUNICORN_PRELOAD", "true").lower() == "true"

# Forward proxy headers (when behind nginx/ALB)
# Default to localhost only; set FORWARDED_ALLOW_IPS=* only if behind a trusted proxy
forwarded_allow_ips = os.environ.get("FORWARDED_ALLOW_IPS", "127.0.0.1")
proxy_allow_ips = os.environ.get("PROXY_ALLOW_IPS", "127.0.0.1")
proxy_protocol = os.environ.get("PROXY_PROTOCOL", "false").lower() == "true"
