import pytest
from src.monitorai._urls import batch_drift_feature

def test_batch_feature_drift():
    batch=batch_drift_feature(variate_type="variate",feature_type="uni")

    assert batch=="drift_feature/batch/variate/uni"
