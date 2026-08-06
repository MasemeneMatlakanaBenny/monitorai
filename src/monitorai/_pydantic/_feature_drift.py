import pandas as pd
from typing import List
from pydantic import BaseModel, ConfigDict, model_validator

class BatchFeatureDriftRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    ref_data: pd.DataFrame
    analysis_data: pd.DataFrame
    feature: str

    @model_validator(mode="after")
    def validate_request(self):
        ref_cols = list(self.ref_data.columns)
        analysis_cols = list(self.analysis_data.columns)

        if ref_cols != analysis_cols:
            raise ValueError(
                "Reference and analysis datasets must have the same columns "
                f"\nReference: {ref_cols}"
                f"\nAnalysis: {analysis_cols}"
            )

        if self.feature not in self.ref_data.columns:
            raise ValueError(
                f"Feature '{self.feature}' not found in reference dataset."
            )

        if self.feature not in self.analysis_data.columns:
            raise ValueError(
                f"Feature '{self.feature}' not found in analysis dataset."
            )

        return self


class BatchMultivariateFeatureDriftRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    ref_data: pd.DataFrame
    analysis_data: pd.DataFrame
    features: List[str]

    @model_validator(mode="after")
    def validate_request(self):
        ref_cols = list(self.ref_data.columns)
        analysis_cols = list(self.analysis_data.columns)

        if ref_cols != analysis_cols:
            raise ValueError(
                "Reference and analysis datasets must have the same columns "
                f"\nReference: {ref_cols}"
                f"\nAnalysis: {analysis_cols}"
            )

        if not self.features:
            raise ValueError("features must contain at least one feature.")

        missing = [
            feature
            for feature in self.features
            if feature not in self.ref_data.columns
        ]

        if missing:
            raise ValueError(
                f"The following features are not present in both datasets: {missing}"
            )

        return self
