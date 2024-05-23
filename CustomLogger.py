import logging


class CustomLogger:
    """
    A class for creating and configuring a custom logger with colored console output.

    This class provides a static method to obtain a logger with a specified name. The logger
    is configured to log messages at the DEBUG level and outputs to the console with custom
    formatting for different log levels.

    Methods:
        getLogger(name: str) -> logging.Logger
            Creates and returns a logger with the specified name, configured with a console handler
            that uses a custom formatter for color-coded log messages.
    """

    @staticmethod
    def getLogger(name: str) -> logging.Logger:
        """
        Creates and returns a logger with the specified name.

        The logger is set to the DEBUG level and is configured with a console handler that
        outputs log messages to the console. The console handler uses `CustomFormatter` to
        format log messages with different colors based on the log level.

        args:
            name : str
                The name of the logger.

        """
        # create logger with the specified name
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # set custom formatter to the console handler
        ch.setFormatter(CustomFormatter())

        # add the console handler to the logger
        logger.addHandler(ch)

        return logger


class CustomFormatter(logging.Formatter):
    """
    A custom formatter for logging that adds color to log messages based on the log level.

    This formatter changes the appearance of log messages for different log levels by adding
    ANSI escape codes for colors. It defines different formats for normal messages and warnings
    or above, including additional information like timestamp, logger name, and source file.

    """

    grey = "\x1b[20;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_normal = "%(message)s"
    format_warning = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    FORMATS = {
        logging.DEBUG: grey + format_normal + reset,
        logging.INFO: grey + format_normal + reset,
        logging.WARNING: yellow + format_warning + reset,
        logging.ERROR: red + format_warning + reset,
        logging.CRITICAL: bold_red + format_warning + reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """
        Formats a log record with color based on the log level.

        This method overrides the default format method to apply color formatting to log
        messages. It selects the appropriate format string from the `FORMATS` dictionary
        based on the log level of the record.

        args:
            record : logging.LogRecord
                The log record to be formatted.
        """
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
