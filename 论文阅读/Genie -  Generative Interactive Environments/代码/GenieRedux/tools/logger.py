import logging
import inspect
import random
from typing import Dict
from enum import Enum


class ANSIColor(Enum):
    BLACK = 0
    RED = 1
    GREEN = 2
    YELLOW = 3
    BLUE = 4
    MAGENTA = 5
    CYAN = 6
    WHITE = 7
    BRIGHT_BLACK = 8
    BRIGHT_RED = 9
    BRIGHT_GREEN = 10
    BRIGHT_YELLOW = 11
    BRIGHT_BLUE = 12
    BRIGHT_MAGENTA = 13
    BRIGHT_CYAN = 14
    BRIGHT_WHITE = 15
    ORANGE = 202
    BRIGHT_ORANGE = 208
    PURPLE = 93
    BRIGHT_PURPLE = 201
    BROWN = 130
    PINK = 201
    BRIGHT_PINK = 213


def map_color(color):
    if isinstance(color, str):
        try:
            return ANSIColor[color.upper()].value
        except KeyError:
            return 0
    return color


def apply_color(message, color):
    """
    Applies the specified color to the given message.

    Args:
        message (str): The message to apply color to.
        color (str): The color name to apply.

    Returns:
        str: The message with the specified color applied.
    """
    return f"\033[38;5;{map_color(color)}m{message}\033[0m"


class ExtendedLogger(logging.Logger):
    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)

    def i(self, *args, **kwargs):
        message = " ".join([str(arg) for arg in args])
        message = apply_color(message, "cyan")
        self.info(message, **kwargs)

    def w(self, *args, **kwargs):
        message = " ".join([str(arg) for arg in args])
        message = apply_color(message, "yellow")
        self.warning(message, **kwargs)

    def e(self, *args, **kwargs):
        message = " ".join([str(arg) for arg in args])
        message = apply_color(message, "red")
        self.error(message, **kwargs)

    def d(self, *args, **kwargs):
        message = " ".join([str(arg) for arg in args])
        self.debug(message, **kwargs)

    def t(self, *args, **kwargs):
        # test message - a debug message shown as info easily found it in code
        message = " ".join([str(arg) for arg in args])
        message = apply_color(message, "green")
        self.info(message, **kwargs)


class CustomFormatter(logging.Formatter):
    def __init__(
        self,
        fmt: str | None = None,
        name_color: int | None = None,
        tag_colors: Dict[str, int] | int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(fmt, **kwargs)
        if tag_colors is None:
            tag_colors = {}

        if name_color is None:
            name_color = random.randint(0, 255)

        self.tag_colors = tag_colors
        self.name_color = name_color

    def format(self, record):
        # Get the stack frame of the calling logger
        stack = inspect.stack()

        # Start from 1 to skip the frame of this function itself
        for frame in stack[1:]:
            classname = frame.frame.f_globals.get("__name__", "unknown")
            if "self" in frame.frame.f_locals:
                classname = frame.frame.f_locals["self"].__class__.__name__

            if classname not in [
                "Logger",
                "StreamHandler",
                "FileHandler",
                "ExtendedLogger",
            ]:
                # Get the class name if available, otherwise use module name

                # Set the color of the classname text
                if isinstance(self.tag_colors, Dict):
                    if classname not in self.tag_colors:
                        self.tag_colors[classname] = random.randint(0, 255)
                    tag_color = self.tag_colors[classname]
                elif isinstance(self.tag_colors, int):
                    tag_color = self.tag_colors
                record.classname = apply_color(classname, tag_color)
                break

        record.name = apply_color(record.name, self.name_color)
        return super(CustomFormatter, self).format(record)


global_file_fpath = None
logging.setLoggerClass(ExtendedLogger)


def getLogger(
    name,
    name_color: int | str | None = None,
    class_colors: Dict[str, int] | str | int | None = None,
) -> ExtendedLogger:
    global global_file_fpath
    # check if logger already exists
    # if name in logging.Logger.manager.loggerDict and name_color is None and class_colors is None:
    #     return logging.getLogger(name)

    logger = logging.getLogger(name)
    logger.propagate = False  # Prevents double logging

    logger.handlers = []

    name_color = map_color(name_color)
    if isinstance(class_colors, Dict):
        for key in class_colors:
            class_colors[key] = map_color(class_colors[key])
    elif isinstance(class_colors, str):
        class_colors = map_color(class_colors)

    formatter = CustomFormatter(
        "[%(asctime)s][%(name)s][%(classname)s] %(levelname)s: %(message)s",
        name_color=name_color,
        tag_colors=class_colors,
        datefmt="%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if global_file_fpath is not None:
        file_handler = logging.FileHandler(global_file_fpath)
        if len(logger.handlers) > 0:
            file_handler.setFormatter(logger.handlers[0].formatter)
            file_handler.setLevel(logger.handlers[0].level)
        logger.addHandler(file_handler)
    return logger


def update_logger_file(log_fpath):
    global global_file_fpath
    for logger in logging.Logger.manager.loggerDict.values():
        if isinstance(logger, logging.Logger):  # Skip placeholders in the loggerDict
            file_handler = logging.FileHandler(log_fpath)
            if len(logger.handlers) > 0:
                file_handler.setFormatter(logger.handlers[0].formatter)
                file_handler.setLevel(logger.handlers[0].level)
            logger.addHandler(file_handler)
    global_file_fpath = log_fpath


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # Logger setup
    logger = getLogger("open-genie", class_colors="BRIGHT_WHITE")

    # Example usage
    class MyClass:
        def my_method(self):
            logger.info("This is an info message")

    # Example usage
    class MyClass2:
        def my_method(self):
            logger.info("This is an info message")

    my_instance = MyClass()
    my_instance.my_method()

    my_instance = MyClass2()
    my_instance.my_method()
