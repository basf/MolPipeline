"""Test logging utils."""

import io
import unittest
from contextlib import redirect_stdout

from molpipeline.utils.logging import print_elapsed_time


class LoggingUtilsTest(unittest.TestCase):
    """Unittest for conversion of sklearn models to json and back."""

    def test__print_elapsed_time(self) -> None:
        """Test message logging with timings work as expected."""

        # when message is None nothing should be printed
        stream1 = io.StringIO()
        with redirect_stdout(stream1):
            with print_elapsed_time("source", message=None, use_logger=False):
                pass
        output1 = stream1.getvalue()
        self.assertEqual(output1, "")

        # message should be printed in the expected sklearn format
        stream2 = io.StringIO()
        with redirect_stdout(stream2):
            with print_elapsed_time("source", message="my message", use_logger=False):
                pass
        output2 = stream2.getvalue()
        self.assertTrue(
            output2.startswith(
                "[source] ................................... my message, total="
            )
        )
