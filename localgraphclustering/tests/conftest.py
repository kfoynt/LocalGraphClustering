import pytest
def pytest_addoption(parser):
    parser.addoption("--test-plots", action="store_true",
                     default=False, help="run test of plotting routines")

    parser.addoption("--long-tests", action="store_true",
                     default=False, help="run longer tests")

    parser.addoption("--notebook-tests", action="store_true",
                     default=False, help="run the tests of the notebooks")

def pytest_runtest_setup(item):
    if 'long_tests' in item.keywords and not item.config.getoption("--long-tests"):
        pytest.skip("need --long-tests option to run this test")
