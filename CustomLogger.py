import logging


class CustomLogger:
    @staticmethod
    def getLogger(name: str):
        # create logger with 'spam_application'
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        ch.setFormatter(CustomFormatter())

        logger.addHandler(ch)
        return logger


class CustomFormatter(logging.Formatter):
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

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


if __name__ == '__main__':
    logger = CustomLogger.getLogger("Test")
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")