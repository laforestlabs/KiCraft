#!/usr/bin/env python3
"""Logging configuration using structlog for fast, structured logging.

Designed for minimal performance impact - logging disabled by default,
only coarse INFO events when enabled, DEBUG for verbose debugging.
"""
from __future__ import annotations
import logging
import os
import threading
from pathlib import Path
from typing import Any

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False


def _get_experiment_dir() -> Path:
    """Get the .experiments directory from environment or default."""
    # Can be overridden via EXPERIMENT_DIR env var
    experiment_dir = os.environ.get("EXPERIMENT_DIR")
    if experiment_dir:
        return Path(experiment_dir)
    # Try to infer from current working directory
    cwd = Path.cwd()
    # Look for .experiments in cwd or parent
    for check in [cwd, cwd.parent, cwd.parent.parent]:
        exp_dir = check / ".experiments"
        if exp_dir.exists():
            return exp_dir
    # Default to .experiments in cwd
    return cwd / ".experiments"


def get_logger(name: str = "autoexperiment") -> Any:
    """Get a configured logger instance.
    
    Usage:
        log = get_logger("autoexperiment")
        log.info("round_started", round_num=5, mode="minor")
        log.info("best_improved", score=87.5, improvement=2.3)
        log.error("experiment_failed", error=str(e))
    """
    if not STRUCTLOG_AVAILABLE:
        # Fallback to stdlib logging
        return _FallbackLogger(name)
    
    return _get_structlog_logger(name)


def _get_structlog_logger(name: str) -> Any:
    """Create a structlog logger with async file writes."""
    experiment_dir = _get_experiment_dir()
    log_file = experiment_dir / "debug.log"
    
    # Ensure log file exists
    experiment_dir.mkdir(parents=True, exist_ok=True)
    if not log_file.exists():
        log_file.touch()
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            _AsyncFileWriter(log_file),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.INFO if os.environ.get("LOG_LEVEL", "").upper() != "DEBUG" else logging.DEBUG
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    return structlog.get_logger(name)


class _AsyncFileWriter:
    """Async file writer that batches writes in background thread.
    
    Non-blocking - writes are queued and flushed periodically.
    """
    
    def __init__(self, log_file: Path, batch_size: int = 10, flush_interval: float = 0.5):
        self.log_file = log_file
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._queue: list[str] = []
        self._lock = threading.Lock()
        self._closed = False
        
        # Start background thread
        self._thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._thread.start()
    
    def __call__(self, logger: Any, method: str, event: dict) -> dict:
        """Add event to queue."""
        with self._lock:
            if self._closed:
                return event
            self._queue.append(self._format_event(event))
            if len(self._queue) >= self.batch_size:
                self._flush()
        return event
    
    def _format_event(self, event: dict) -> str:
        """Format event as single line."""
        # Extract key fields
        timestamp = event.get("timestamp", "")
        level = event.get("level", "info")
        event_name = event.get("event", "")
        
        # Remove standard fields, keep rest as key=value pairs
        extras = {k: v for k, v in event.items() 
                  if k not in ("timestamp", "level", "event")}
        
        if extras:
            extras_str = " ".join(f"{k}={v}" for k, v in extras.items())
            return f"[{timestamp}] {level.upper()}: {event_name} ({extras_str})\n"
        return f"[{timestamp}] {level.upper()}: {event_name}\n"
    
    def _flush(self):
        """Flush queue to file."""
        if not self._queue:
            return
        
        to_write = self._queue
        self._queue = []
        
        try:
            with open(self.log_file, "a") as f:
                f.writelines(to_write)
        except OSError:
            pass  # Non-fatal
    
    def _flush_loop(self):
        """Background flush loop."""
        import time
        while not self._closed:
            time.sleep(self.flush_interval)
            with self._lock:
                if not self._closed:
                    self._flush()
    
    def close(self):
        """Close and flush remaining."""
        with self._lock:
            self._closed = True
            self._flush()


class _FallbackLogger:
    """Fallback logger using stdlib logging when structlog unavailable."""
    
    def __init__(self, name: str):
        self.name = name
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.INFO)
        
        # Stream handler if not already configured
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            )
            self._logger.addHandler(handler)
    
    def debug(self, event: str, **kwargs):
        self._logger.debug(event, extra=kwargs)
    
    def info(self, event: str, **kwargs):
        self._logger.info(event, extra=kwargs)
    
    def warning(self, event: str, **kwargs):
        self._logger.warning(event, extra=kwargs)
    
    def error(self, event: str, **kwargs):
        self._logger.error(event, extra=kwargs)
    
    def exception(self, event: str, **kwargs):
        self._logger.exception(event, extra=kwargs)


def configure_logging(log_level: str = "INFO", experiment_dir: str = None):
    """Configure logging from main entry point.
    
    Args:
        log_level: "INFO" or "DEBUG"
        experiment_dir: Path to .experiments directory
    """
    if experiment_dir:
        os.environ["EXPERIMENT_DIR"] = experiment_dir
    
    os.environ["LOG_LEVEL"] = log_level.upper()
    
    if log_level.upper() == "DEBUG":
        # Enable debug logging
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )


# Decorator for adding log context
def with_log_context(**context):
    """Decorator to add context to logging calls.
    
    Usage:
        @with_log_context(component="U1", net="VCC")
        def route_net(net):
            log.info("routing_net", net_length=len(path))
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Add context for this call
            for key, value in context.items():
                structlog.contextvars.clear_contextvars()
                structlog.contextvars.bind_contextvars(**{key: value})
            return func(*args, **kwargs)
        return wrapper
    return decorator