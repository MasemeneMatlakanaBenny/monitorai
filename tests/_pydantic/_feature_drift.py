import pandas as pd
import pytest
from pydantic import ValidationError
from src.monitorai._pydantic import BatchFeatureDriftRequest,BatchMultivariateFeatureDriftRequest



@pytest.fixture
def ref_df():
    return pd.DataFrame(
        {
            "age": [20, 30, 40],
            "salary": [1000, 2000, 3000],
            "city": ["A", "B", "C"],
        }
    )


@pytest.fixture
def analysis_df():
    return pd.DataFrame(
        {
            "age": [25, 35, 45],
            "salary": [1200, 2200, 3200],
            "city": ["D", "E", "F"],
        }
    )


@pytest.fixture
def analysis_df_bad_columns():
    return pd.DataFrame(
        {
            "age": [25, 35, 45],
            "income": [1200, 2200, 3200],
            "city": ["D", "E", "F"],
        }
    )

def test_batch_request_valid(ref_df, analysis_df):
    request = BatchFeatureDriftRequest(
        ref_data=ref_df,
        analysis_data=analysis_df,
        feature="age",
    )

    assert request.feature == "age"


def test_batch_request_raises_for_mismatched_columns(
    ref_df,
    analysis_df_bad_columns,
):
    with pytest.raises(ValidationError, match="same columns"):
        BatchFeatureDriftRequest(
            ref_data=ref_df,
            analysis_data=analysis_df_bad_columns,
            feature="age",
        )


def test_batch_request_raises_for_missing_feature(
    ref_df,
    analysis_df,
):
    with pytest.raises(ValidationError, match="not found"):
        BatchFeatureDriftRequest(
            ref_data=ref_df,
            analysis_data=analysis_df,
            feature="height",
        )




def test_multivariate_request_valid(ref_df, analysis_df):
    request = BatchMultivariateFeatureDriftRequest(
        ref_data=ref_df,
        analysis_data=analysis_df,
        features=["age", "salary"],
    )

    assert request.features == ["age", "salary"]


def test_multivariate_request_raises_for_mismatched_columns(
    ref_df,
    analysis_df_bad_columns,
):
    with pytest.raises(ValidationError, match="same columns"):
        BatchMultivariateFeatureDriftRequest(
            ref_data=ref_df,
            analysis_data=analysis_df_bad_columns,
            features=["age"],
        )


def test_multivariate_request_raises_for_missing_features(
    ref_df,
    analysis_df,
):
    with pytest.raises(
        ValidationError,
        match="not present in both datasets",
    ):
        BatchMultivariateFeatureDriftRequest(
            ref_data=ref_df,
            analysis_data=analysis_df,
            features=["age", "height"],
        )


def test_multivariate_request_raises_for_empty_features(
    ref_df,
    analysis_df,
):
    with pytest.raises(
        ValidationError,
        match="at least one feature",
    ):
        BatchMultivariateFeatureDriftRequest(
            ref_data=ref_df,
            analysis_data=analysis_df,
            features=[],
        )
