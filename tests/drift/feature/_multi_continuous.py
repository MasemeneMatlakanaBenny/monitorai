import pytest
import os
import pandas as pd
from src.monitorai import MonitorAI
from src.monitorai.drift.feature import MultivariateContinuousFeatureDrift


def feature_drift()->MultivariateContinuousFeatureDrift:
    api_key=os.getenv("MonitorAI_API_KEY")
    client=MonitorAI(api_key)
    ref_df=pd.DataFrame({
            "temperature":[11,14,15,37,55,88,90,41,19,19],
            "prices":[15,18,43,55,67,91,110,21,39,81]
        })
    
    
    ana_df=pd.DataFrame({
        "temperature":[190,93,300,600,704,703,800,1700],
        "prices":[400,560,77,98,165,55,98,980]
    })

    return MultivariateContinuousFeatureDrift(client=client,
                                ref_data=ref_df,
                                analysis_data=ana_df)


def test_bhatta_coeff():
    cont_drift=feature_drift()
    bhatta_coeff=cont_drift.bhatta_coeff(features=["temperature","price"])
   
    assert isinstance(bhatta_coeff,float)
  
def test_euclidean_dist():
    cont_drift=feature_drift()
    euclid_dist=cont_drift.euclidean_dist(features=["temperature","price"])
 
    assert isinstance(euclid_dist,float)

def test_cauchy_schwartz_dist():
    cont_drift=feature_drift()
    cs_dist=cont_drift.cauchy_schwartz_dist(features=["temperature","price"])

    assert isinstance(cs_dist,float)
   
def test_hellinger_dist():
    cont_drift=feature_drift()
    hellinger_dist=cont_drift.hellinger_dist(features=["temperature","price"])

    assert isinstance(hellinger_dist,float)

def test_js_div():
    cont_drift=feature_drift()
    js_div=cont_drift.js_divergence(features=["temperature","price"])

    assert isinstance(js_div,float)
   
def test_kl_div():
    cont_drift=feature_drift()
    kl_div=cont_drift.kl_divergence(features=["temperature","price"])

    assert isinstance(kl_div,float)
    assert kl_div>=0
  
def test_psi():
    cont_drift=feature_drift()
    psi=cont_drift.psi(features=["temperature","price"])
   
    assert isinstance(psi,float)
    assert psi>=0

def test_tvd():
    cont_drift=feature_drift()
    tvd=cont_drift.total_variation_dist(features=["temperature","price"])

    assert isinstance(tvd,float)
  
def test_wasserstein():
    cont_drift=feature_drift()
    wass_dist=cont_drift.wasserstein_dist(features=["temperature","price"])
   
    assert isinstance(wass_dist,float)
    


def test_with_same_data()->MultivariateContinuousFeatureDrift:
    api_key=os.getenv("MonitorAI_API_KEY")
    client=MonitorAI(api_key)
    ref_df=pd.DataFrame({
                "temperature":[11,14,15,37,55,88,90,41,19,19],
                "prices":[15,18,43,55,67,91,110,21,39,81]
            })


    return MultivariateContinuousFeatureDrift(client=client,
                                ref_data=ref_df,
                                analysis_data=ref_df)

def test_same_bhatta_coeff():
    cont_drift=test_with_same_data()
    bhatta_coeff=cont_drift.bhatta_coeff(features=["temperature","price"])
   
    assert isinstance(bhatta_coeff,float)
    assert bhatta_coeff==0
  
def test_same_euclidean_dist():
    cont_drift=test_with_same_data()
    euclid_dist=cont_drift.euclidean_dist(features=["temperature","price"])
 
    assert isinstance(euclid_dist,float)
    assert euclid_dist==0

def test_same_cauchy_schwartz_dist():
    cont_drift=test_with_same_data()
    cs_dist=cont_drift.cauchy_schwartz_dist(features=["temperature","price"])

    assert isinstance(cs_dist,float)
    assert cs_dist==0
   
def test_same_hellinger_dist():
    cont_drift=test_with_same_data()
    hellinger_dist=cont_drift.hellinger_dist(features=["temperature","price"])

    assert isinstance(hellinger_dist,float)
    assert hellinger_dist==0

def test_same_js_div():
    cont_drift=test_with_same_data()
    js_div=cont_drift.js_divergence(features=["temperature","price"])

    assert isinstance(js_div,float)
    assert js_div==0
   
def test_same_kl_div():
    cont_drift=test_with_same_data()
    kl_div=cont_drift.kl_divergence(features=["temperature","price"])

    assert isinstance(kl_div,float)
    assert kl_div==0
  
def test_same_psi():
    cont_drift=test_with_same_data()
    psi=cont_drift.psi(features=["temperature","price"])
   
    assert isinstance(psi,float)
    assert psi==0

def test_same_tvd():
    cont_drift=test_with_same_data()
    tvd=cont_drift.total_variation_dist(features=["temperature","price"])

    assert isinstance(tvd,float)
    assert tvd==0
  
def test_same_wasserstein():
    cont_drift=test_with_same_data()
    wass_dist=cont_drift.wasserstein_dist(features=["temperature","price"])
   
    assert isinstance(wass_dist,float)
    assert wass_dist==0
    
