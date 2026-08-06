import requests
import pandas as pd
from src.monitorai import MonitorAI
from _pydantic import BatchFeatureDriftRequest
from _urls import batch_drift_feature
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



class CategoricalFeatureDrift:
    def __init__(self,
                 client:MonitorAI,
                 ref_data:pd.DataFrame,
                 analysis_data:pd.DataFrame):
        """
        The CategoricalFeatureDrift class is used to calculate drift for features between the reference and analysis datasets.
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

    def _response(self,feature:str):
        try:

            drift_request=BatchFeatureDriftRequest(
                ref_data=self.ref_data,
                analysis_data=self.analysis_data,
                feature=feature
            )
            response=requests.post(
                url=f"{self.client.base_url}/{batch_drift_feature(variate_type="univariate",
                                                              feature_type="categorical")}",
                headers=self.client.headers,
                json=drift_request.model_dump()
            )

        except requests.exceptions.ConnectionError:
            raise NetworkError("Failed network connection")

        except requests.exceptions.Timeout:
            raise TimeoutError("Request time out")

        if requests.status_codes==400:
            raise BadRequestError("Request Error")

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
            raise ServerError("Internal server failed.")

        
        elif not response.ok:
            raise MonitorAIError(response.text)
        
        response.raise_for_status()

        return response.json()

    
    def bhatta_coeff(self,feature:str):
        """
        Args:
            feature (str) : the feature to compute drift using the bhattacharrya coefficient for.

        Returns:
            float: the bhattacharrya coefficient
        """
        return self._response(feature)["bhatta_coef"]

    def cauchy_schwartz_dist(self,feature:str):
        """
        Args:
            feature (str) : the feature to compute drift using the cauchy-schwartz distance.
        
        Returns:
            float: the bhattacharrya coefficient
        """

        return self._response(feature=feature)["cauchy_schwartz"]

    def euclidean_dist(self,feature:str):
        """
        Args:
            feature (str) : the feature to compute drift for using the euclidean distance.

        Returns:
            float: the euclidean distance.
        """
        return self._response(feature)["euclidean_dist"]

    def hellinger_dist(self,feature:str):
        """
        Args:
            feature (str) : the feature to compute drift for using the hellinger distance.

        Returns:
            float : the hellinger distance.
        """
        return self._response(feature)["hellinger_dist"]

    def js_divergence(self,feature:str):
        """
        Args:
            feature (str) : the feature to compute drift for using jensen shannon divergence metric.

        Returns:
            float: the jensen shannon divergence
        """
        return self._response(feature)["js_divergence"]

    def kl_divergence(self,feature:str):
        """
        Args:
            feature (str) : the feature to compute drift for using kullback-leibler divergence metric.

        Returns:
            float: the kullback leibler divergence
        """
        return self._response(feature)["kl_divergence"]

    def psi(self,feature:str):
        """
        Args:
            feature (str) : the feature to compute drift for using population stability index method.

        Returns:
            float: the population stability index 
        """
        return self._response(feature)["psi"]

    def total_variation_dist(self,feature:str):
        """
        Args:
            feature (str) : the feature compute drift for using the total variation drift

        Returns:
            float : the total variation distance
        """
        return self._response(feature)["total_variation_dist"]

