import requests
import pandas as pd
from typing import List
from src.monitorai import MonitorAI
from _urls import batch_drift_feature
from _pydantic import BatchMultivariateFeatureDriftRequest
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

class MultivariateContinuousFeatureDrift:
    def __init__(self,
                 client:MonitorAI,
                 ref_data:pd.DataFrame,
                 analysis_data:pd.DataFrame):

        """
        The MultivariateContinuousFeatureDrift class is used to calculate drift for features between the reference and analysis datasets.
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

    def _response(self,features):
        try:
            drift_request=BatchMultivariateFeatureDriftRequest(ref_data=self.ref_data,
                                                               analysis_data=self.analysis_data,
                                                               features=features)
            response=requests.get(
                url=f"{self.client.base_url}/{batch_drift_feature(
                    variate_type="multivariate",
                    feature_type="continuous"
                )}",
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

    def bhatta_coeff(self,features:List[str]):
        """
        Args:
            features (List[str]) : the list of features to compute drift using the bhattacharrya coefficient for.
        
        Returns:
            float: the bhattacharrya coefficient
        """
        return self._response(features)["bhatta_coeff"]

    def cauchy_schwartz_dist(self,features:List[str]):
        """
        Args:
            features (List[str]) : the list of features to compute drift using the cauchy-schwartz distance.
        
        Returns:
            float: the bhattacharrya coefficient.
        """
        return self._response(features=features)["cauchy_schwartz"]


    def distance_cov(self,features:List[str]):
        """
        Args:
            features (List[str]) : the list of features to compute drift using the distance covariance.
                
        Returns:
            float: the distance covariance.
        """
        return self._response(features)["distance_cov"]
    
    def distance_corr(self,features:List[str]):
        """
        Args:
            features (List[str]) : the list of features to compute correlation using the distance correlation.
            
        Returns:
            float: the distance correlation.
        """
        return self._response(features)["distance_corr"]
    
    def euclidean_dist(self,features:List[str]):
        """
        Args:
            features (List[str]) : the list of features to compute drift for using the euclidean distance.
    
        Returns:
            float: the euclidean distance.
        """
        return self._response(features)["euclidean_dist"]

    def hellinger_dist(self,features:List[str]):
        """
        Args:
            features (List[str]) : the list of features to compute drift for using the hellinger distance.

        Returns:
            float : the hellinger distance.
        """
        return self._response(features)["hellinger_dist"]

    def js_divergence(self,features:List[str]):
        """
        Args:
            features (List[str]) : the list of features to compute drift for using jensen shannon divergence metric.
        
        Returns:
            float: the jensen shannon divergence.
        """
        return self._response(features)["js_divergence"]

    def kl_divergence(self,features:List[str]):
        """
        Args:
            features (List[str]) : the list of features to compute drift for using kullback-leibler divergence metric.
        
        Returns:
            float: the kullback leibler divergence.
        """
        return self._response(features)["kl_divergence"]

    def mmd_stat(self,features:List[str]):
        """
        Args:
            features (List[str]) : the feature to compute drift using the maximum mean disperancy.
                
        Returns:
            float: the maximum mean disperancy.
             
        """

        return self._response(features)["mmd_stat"]
    
    def std_e_stat(self,features:List[str]):
        """
        Args:
            features (List[str]) : the list of features to compute the standardized energy statistic.
                
        Returns:
            float: the standardized energy statistic.
        """
        return self._response(features)["std_energy_stat"]

    def psi(self,features:List[str]):
        """
        Args:
            features (List[str]) : the list of features to compute drift for using population stability index method.

        Returns:
            float: the population stability index,
        """
        
        return self._response(features)["psi"]

    def total_variation_dist(self,features:List[str]):
        """
        Args:
            feature (List[str]) : the list of features compute drift for using the wasserstein distance.
        
        Returns:
            float : the total variation distance.
        """
        return self._response(features)["total_variation_dist"]

    def wasserstein_dist(self,features:List[str]):
        """
        Args:
            features (List[str]) : the list of features compute drift for using the wasserstein distance.
        
        Returns:
            float : the wasserstein distance.
        """
        return self._response(features)["wasserstein_dist"]



