import logging
import sys


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Настраивает логгер с форматированием"""
    logger = logging.getLogger(name)

    log_level = getattr(logging, level.upper())
    logger.setLevel(log_level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger