import unittest
from unittest.mock import patch
from io import StringIO
from logger import getLogger, Color


class LoggerTest(unittest.TestCase):
    def setUp(self):
        self.logger = getLogger(
            "test_logger",
            name_color=Color.BRIGHT_YELLOW,
            class_colors={"MyClass": Color.BRIGHT_CYAN},
        )

    def test_logger_output(self):
        with patch("sys.stdout", new=StringIO()) as fake_stdout:
            self.logger.info("Test message")
            self.assertEqual(
                fake_stdout.getvalue().strip(),
                "\x1b[38;5;11m[test_logger]\x1b[0m \x1b[38;5;14m[MyClass]\x1b[0m INFO: Test message",
            )

    def test_logger_multiple_instances(self):
        logger1 = getLogger("logger1", name_color=Color.BRIGHT_RED)
        logger2 = getLogger("logger2", name_color=Color.BRIGHT_GREEN)

        with patch("sys.stdout", new=StringIO()) as fake_stdout:
            logger1.info("Message from logger1")
            logger2.info("Message from logger2")

            self.assertEqual(
                fake_stdout.getvalue().strip(),
                "\x1b[38;5;9m[logger1]\x1b[0m INFO: Message from logger1\n\x1b[38;5;10m[logger2]\x1b[0m INFO: Message from logger2",
            )


if __name__ == "__main__":
    unittest.main()
