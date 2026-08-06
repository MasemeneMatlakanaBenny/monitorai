import requests
import pandas as pd
from src.monitorai import MonitorAI
from _urls import batch_drift_feature
from _pydantic import BatchFeatureDriftRequest
from _exceptions import (
    AuthenticationError,
    BadRequestError,
    ConflictError,
    NetworkError,
    NotFoundError,
    MonitorAIError,
    PermissionDeniedError,
    RateLimitError,
    ServerError,
    TimeoutError,
    ValidationError)

class SimilarityCategoricalFeatureDrift:
    def __init__(self,
                 client:MonitorAI,
                 ref_data:pd.DataFrame,
                 analysis_data:pd.DataFrame):
        """
        The SimilarityCategoricalFeatureDrift class is used to calculate drift for features between the reference and analysis datasets.
        Args:
            client (MonitorAI):
                the client for monitorai.
    
            ref_data (pandas.DataFrame):
                the reference pandas dataframe.
    
            analysis_data(pandas.DataFrame):
                the analysis pandas dataframe.
        """
        self.client=client
        self.ref_data=ref_data
        self.analysis_data=analysis_data


    def _response(self,feature):
        try:
            drift_request=BatchFeatureDriftRequest(ref_data=self.ref_data,
                                                    analysis_data=self.analysis_data,
                                                    feature=feature)
            response=requests.post(
                url=f"{self.client.base_url}/{batch_drift_feature(variate_type="univariate",
                                                              feature_type="similarity")}",
                headers=self.client.headers,
              json=drift_request.model_dump()
            )

        except requests.exceptions.ConnectionError:
            raise NetworkError("Failed network connection")
                
        except requests.exceptions.Timeout:
            raise TimeoutError("Request time out")
                
        if requests.status_codes==400:
            raise BadRequestError("")
                
        elif response.status_code == 401:
            raise AuthenticationError("Invalid API key.")
                
        elif response.status_code==403:
            raise PermissionDeniedError("Permission denied")
                
        elif response.status_code==404:
            raise NotFoundError("Invalid request")
                
        elif response.status_code==409:
            raise ConflictError("conflict error")
                
        elif response.status_code == 422:
            raise ValidationError(response.text)
                        
        elif response.status_code == 429:
            raise RateLimitError("Rate limit exceeded.")
                
        elif response.status_code >= 500:
            raise ServerError("Calculator server failed.")
                
        elif not response.ok:
            raise MonitorAIError(response.text)
        
        response.raise_for_status()

        return response.json()

    def braun_coeff(self,feature:str):
        """
        Args:
            feature (str) : the feature to compute drift for using braun coefficient

        Returns:
            float : the braun coefficient
        """
        return self._response(feature)["braun_coeff"]

    def dice_coeff(self,feature:str):
        """
        Args:
            feature (str) : the feature to compute drift for using the dice coefficient

        Returns:
            float : the dice coefficient
        """
        return self._response(feature)["dice_coeff"]

    def jaccard_coeff(self,feature:str):
        """
        Args:
            feature (str) : the feature to compute drift for using the jaccard coefficient

        Returns:
            float : the jaccard coefficient
        """
        return self._response(feature)["jaccard_coeff"]

    def overlap_coeff(self,feature:str):
        """
        Args:
            feature (str): the feature to compute drift for using the overlap coefficient

        Returns:
            float: the overlap coefficient
        """
        return self._response(feature)["overlap_coeff"]

    def tanimoto_coeff(self,feature:str):
        """
        Args:
            feature (str): the feature to compute drift for using the tanimoto coefficient

        Returns:
            float : the tanimoto coefficient
        """
        return self._response(feature)["tanimoto_coeff"]


