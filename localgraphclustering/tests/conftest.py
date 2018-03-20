import pytest
def pytest_addoption(parser):
    parser.addoption("--test-plots", action="store_true",
                     default=False, help="run test of plotting routines")
