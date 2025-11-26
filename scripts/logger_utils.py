#!/usr/bin/env python
"""
Logging utilities for all scripts in the pipeline.

Each script will create its own log file in logs/ directory with:
- Timestamp in filename for history
- Real-time console output
- File output for persistence
- Support for tail -f monitoring
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(script_name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Setup logger with both file and console handlers.

    Args:
        script_name: Name of the script (e.g., 'build_dataset', 'train', 'eval')
        log_dir: Directory to store log files (default: 'logs')

    Returns:
        Configured logger instance

    Log file format: logs/<script_name>_YYYYMMDD_HHMMSS.log
    Example: logs/train_20250127_143022.log

    Usage:
        from logger_utils import setup_logger
        logger = setup_logger('train')
        logger.info('Training started')
    """
    # Create logs directory if not exists
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Create timestamp for log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{script_name}_{timestamp}.log"

    # Create logger
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # File handler - write everything to file
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler - show INFO and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Log the log file location
    logger.info(f"Logging to: {log_file}")
    logger.info(f"Monitor progress: tail -f {log_file}")

    return logger


def log_section(logger: logging.Logger, title: str):
    """Log a section separator for better readability."""
    separator = "=" * 60
    logger.info(separator)
    logger.info(title)
    logger.info(separator)


def log_config(logger: logging.Logger, config: dict):
    """Log configuration parameters."""
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")


def log_metrics(logger: logging.Logger, metrics: dict, prefix: str = ""):
    """Log metrics in a formatted way."""
    if prefix:
        logger.info(f"{prefix}:")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")


class TqdmLoggingHandler(logging.Handler):
    """
    Custom logging handler that works well with tqdm progress bars.
    Prevents logging from interfering with progress bar display.
    """
    def emit(self, record):
        try:
            from tqdm import tqdm
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            # Fallback to standard output if tqdm not available
            print(self.format(record))


def setup_logger_with_tqdm(script_name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Setup logger that works nicely with tqdm progress bars.

    Same as setup_logger but console output goes through tqdm.write()
    to avoid interfering with progress bars.
    """
    # Create logs directory if not exists
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Create timestamp for log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{script_name}_{timestamp}.log"

    # Create logger
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # File handler - write everything to file
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler with tqdm support
    console_handler = TqdmLoggingHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Log the log file location
    logger.info(f"Logging to: {log_file}")
    logger.info(f"Monitor progress: tail -f {log_file}")

    return logger
