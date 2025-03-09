"""
Logger module for the Daymenion AI Suite.

This module provides a centralized logging system for the entire application.
It supports console and file logging with configurable log levels and formats.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import datetime
import traceback
from typing import Optional, Union, Dict, Any

# Import other common modules
from common.config import LOG_SETTINGS


# Function moved from utils to avoid circular imports
def format_timestamp(timestamp: Optional[datetime.datetime] = None) -> str:
    """
    Format a timestamp in a standardized way for logs and filenames.
    
    Args:
        timestamp: Datetime to format, or current time if None
        
    Returns:
        str: Formatted timestamp string
    """
    if timestamp is None:
        timestamp = datetime.datetime.now()
    return timestamp.strftime("%Y%m%d_%H%M%S")


# Function to create directory without importing from utils
def ensure_directory(directory_path: str) -> str:
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        str: Path to the created directory
    """
    os.makedirs(directory_path, exist_ok=True)
    return directory_path


# Default log settings if not in config
DEFAULT_LOG_SETTINGS = {
    "log_level": "INFO",
    "console_log_level": "INFO",
    "file_log_level": "DEBUG",
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
    "log_dir": "logs",
    "app_log_file": "app.log",
    "max_log_size_mb": 10,
    "backup_count": 5,
    "module_specific_levels": {
        "nerd_ai": "INFO",
        "interior_design": "INFO",
        "music_generator": "INFO",
        "frontend": "INFO",
        "common": "INFO"
    }
}


class AppLogger:
    """
    Centralized logging system for the Daymenion AI Suite.
    
    This class provides a singleton logger instance that can be used
    across the entire application, ensuring consistent logging behavior.
    """
    
    _instance = None
    _initialized = False
    _loggers = {}
    
    def __new__(cls, *args, **kwargs):
        """Ensure singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super(AppLogger, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the logger if not already initialized."""
        if not self._initialized:
            self._setup_logging()
            self._initialized = True
    
    def _setup_logging(self):
        """Set up the logging configuration."""
        # Get log settings from config or use defaults
        self.settings = getattr(LOG_SETTINGS, "LOG_SETTINGS", DEFAULT_LOG_SETTINGS)
        
        # Create log directory if it doesn't exist
        self.log_dir = self.settings.get("log_dir", DEFAULT_LOG_SETTINGS["log_dir"])
        ensure_directory(self.log_dir)
        
        # Set up root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)  # Capture everything, handlers will filter
        
        # Remove existing handlers to avoid duplicates on re-initialization
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Create formatters
        log_format = self.settings.get("log_format", DEFAULT_LOG_SETTINGS["log_format"])
        date_format = self.settings.get("date_format", DEFAULT_LOG_SETTINGS["date_format"])
        formatter = logging.Formatter(log_format, date_format)
        
        # Set up console handler
        console_level = getattr(logging, self.settings.get("console_log_level", 
                                                         DEFAULT_LOG_SETTINGS["console_log_level"]))
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # Set up file handler with rotation
        file_level = getattr(logging, self.settings.get("file_log_level", 
                                                       DEFAULT_LOG_SETTINGS["file_log_level"]))
        log_file = os.path.join(self.log_dir, self.settings.get("app_log_file", 
                                                             DEFAULT_LOG_SETTINGS["app_log_file"]))
        max_bytes = self.settings.get("max_log_size_mb", DEFAULT_LOG_SETTINGS["max_log_size_mb"]) * 1024 * 1024
        backup_count = self.settings.get("backup_count", DEFAULT_LOG_SETTINGS["backup_count"])
        
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=max_bytes, 
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # Initialize the 'app' logger
        self._loggers['app'] = root_logger
        
        # Log startup message
        self.info("Logging system initialized", "app")
    
    def get_logger(self, module_name: str) -> logging.Logger:
        """
        Get a logger for a specific module with appropriate log level.
        
        Args:
            module_name: The name of the module requesting the logger
            
        Returns:
            logging.Logger: A configured logger instance
        """
        if module_name in self._loggers:
            return self._loggers[module_name]
        
        # Create a new logger for this module
        logger = logging.getLogger(module_name)
        
        # Set module-specific log level if configured
        module_levels = self.settings.get("module_specific_levels", 
                                        DEFAULT_LOG_SETTINGS["module_specific_levels"])
        
        # Find the most specific module level setting
        matched_module = None
        for configured_module in module_levels:
            if module_name.startswith(configured_module) and (
                matched_module is None or len(configured_module) > len(matched_module)
            ):
                matched_module = configured_module
        
        if matched_module:
            level_name = module_levels[matched_module]
            level = getattr(logging, level_name)
            logger.setLevel(level)
        
        # Store the logger for reuse
        self._loggers[module_name] = logger
        
        return logger
    
    def _log(self, level: int, message: str, module: str, exc_info=None, extra=None):
        """Internal method to log a message with the specified level."""
        logger = self.get_logger(module)
        logger.log(level, message, exc_info=exc_info, extra=extra)
    
    def debug(self, message: str, module: str = "app", extra: Dict[str, Any] = None):
        """Log a debug message."""
        self._log(logging.DEBUG, message, module, extra=extra)
    
    def info(self, message: str, module: str = "app", extra: Dict[str, Any] = None):
        """Log an info message."""
        self._log(logging.INFO, message, module, extra=extra)
    
    def warning(self, message: str, module: str = "app", extra: Dict[str, Any] = None):
        """Log a warning message."""
        self._log(logging.WARNING, message, module, extra=extra)
    
    def error(self, message: str, module: str = "app", exc_info=None, extra: Dict[str, Any] = None):
        """Log an error message, optionally with exception info."""
        self._log(logging.ERROR, message, module, exc_info=exc_info, extra=extra)
    
    def critical(self, message: str, module: str = "app", exc_info=None, extra: Dict[str, Any] = None):
        """Log a critical message, optionally with exception info."""
        self._log(logging.CRITICAL, message, module, exc_info=exc_info, extra=extra)
    
    def exception(self, message: str, module: str = "app", extra: Dict[str, Any] = None):
        """Log an exception message with stack trace."""
        self._log(logging.ERROR, message, module, exc_info=True, extra=extra)
    
    def log_exception(self, e: Exception, module: str = "app", level: str = "ERROR", message: str = None):
        """
        Log an exception with the specified level and optional custom message.
        
        Args:
            e: The exception to log
            module: The module name
            level: Log level as string ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
            message: Optional custom message to include with the exception
        """
        log_message = message if message else f"Exception occurred: {str(e)}"
        
        log_level = getattr(logging, level.upper())
        self._log(log_level, log_message, module, exc_info=e)
    
    @staticmethod
    def get_instance():
        """Get the singleton instance of the AppLogger."""
        if AppLogger._instance is None:
            return AppLogger()
        return AppLogger._instance


def get_logger(module_name: str = "app") -> AppLogger:
    """
    Get the application logger with a specified module name.
    
    This is the recommended way to access the logging system from any module.
    
    Args:
        module_name: The name of the module using the logger
        
    Returns:
        AppLogger: The configured logger instance
    """
    return AppLogger.get_instance()


# Create a convenience function for logging exceptions in a context manager
class LogExceptionContext:
    """
    Context manager for logging exceptions.
    
    Example:
        with LogExceptionContext("some_operation", "nerd_ai"):
            # code that might raise an exception
            result = potentially_failing_function()
    """
    
    def __init__(self, operation: str, module: str = "app", level: str = "ERROR"):
        """
        Initialize the context manager.
        
        Args:
            operation: Description of the operation being performed
            module: The module name
            level: Log level as string ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        """
        self.operation = operation
        self.module = module
        self.level = level
        self.logger = get_logger()
    
    def __enter__(self):
        """Enter the context."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context, logging any exception that occurred.
        
        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback
            
        Returns:
            bool: True to suppress the exception, False to propagate it
        """
        if exc_type is not None:
            # An exception occurred
            log_level = getattr(logging, self.level.upper())
            self.logger._log(
                log_level,
                f"Error during {self.operation}: {exc_val}",
                self.module,
                exc_info=(exc_type, exc_val, exc_tb)
            )
        
        # Don't suppress the exception
        return False


# Helper function for using the exception context
def log_exceptions(operation: str, module: str = "app", level: str = "ERROR"):
    """
    Create a context manager for logging exceptions.
    
    Example:
        with log_exceptions("processing_image", "nerd_ai"):
            processed_image = process_image(raw_image)
    
    Args:
        operation: Description of the operation being performed
        module: The module name
        level: Log level as string ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        
    Returns:
        LogExceptionContext: A context manager for exception logging
    """
    return LogExceptionContext(operation, module, level) 