"""Numerical regression tests for the AquaPose PosePipeline.

These tests run the new PosePipeline on the golden-data clip and compare
every stage's output to committed golden reference data in tests/golden/.

All tests are marked @pytest.mark.regression and excluded from the fast
test loop. Run them with:

    hatch run test-regression

or:

    pytest tests/regression/ -m regression
"""
