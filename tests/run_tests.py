"""Run all tests."""

import unittest

if __name__ == "__main__":
    pipeline_test = unittest.TestLoader().discover(".")
    unittest.TextTestRunner(verbosity=2).run(pipeline_test)
