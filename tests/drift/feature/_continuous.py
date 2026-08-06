import pytest
import os
import pandas as pd
from src.monitorai import MonitorAI
from src.monitorai.drift.feature import ContinuousFeatureDrift


def feature_drift()->ContinuousFeatureDrift:
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

    return ContinuousFeatureDrift(client=client,
                                ref_data=ref_df,
                                analysis_data=ana_df)


def test_bhatta_coeff():
    cont_drift=feature_drift()
    bhatta_temp=cont_drift.bhatta_coeff(feature="temperature")
    bhatta_prices=cont_drift.bhatta_coeff(feature="prices")


    assert isinstance(bhatta_temp,float)
    assert isinstance(bhatta_prices,float)


def test_euclidean_dist():
    cont_drift=feature_drift()
    euclid_temp=cont_drift.euclidean_dist("temperature")
    euclid_prices=cont_drift.euclidean_dist("prices")

    assert isinstance(euclid_temp,float)
    assert isinstance(euclid_prices,float)

def test_cauchy_schwartz_dist():
    cont_drift=feature_drift()
    cs_dist_temp=cont_drift.cauchy_schwartz_dist("temperature")
    cs_dist_prices=cont_drift.cauchy_schwartz_dist("prices")

    assert isinstance(cs_dist_temp,float)
    assert isinstance(cs_dist_prices,float)


def test_hellinger_dist():
    cont_drift=feature_drift()
    hellinger_temp=cont_drift.hellinger_dist("temperature")
    hellinger_prices=cont_drift.hellinger_dist("prices")

    assert isinstance(hellinger_temp,float)
    assert isinstance(hellinger_prices,float)

def test_js_div():
    cont_drift=feature_drift()
    js_temp=cont_drift.js_divergence("temperature")
    js_prices=cont_drift.js_divergence("prices")

    assert isinstance(js_temp,float)
    assert isinstance(js_prices,float)

def test_kl_div():
    cont_drift=feature_drift()
    kl_temp=cont_drift.kl_divergence("temperature")
    kl_prices=cont_drift.kl_divergence("prices")

    assert isinstance(kl_temp,float)
    assert isinstance(kl_prices,float)

def test_psi():
    cont_drift=feature_drift()
    psi_temp=cont_drift.psi("temperature")
    psi_prices=cont_drift.psi("prices")

    assert isinstance(psi_temp,float)
    assert isinstance(psi_prices,float)

    assert psi_temp>=0
    assert psi_prices>=0

def test_tvd():
    cont_drift=feature_drift()
    tvd_temp=cont_drift.total_variation_dist("temperature")
    tvd_prices=cont_drift.total_variation_dist("prices")

    assert isinstance(tvd_temp,float)
    assert isinstance(tvd_prices,float)


def test_wasserstein():
    cont_drift=feature_drift()
    wass_temp=cont_drift.wasserstein_dist("temperature")
    wass_prices=cont_drift.wasserstein_dist("prices")

    assert isinstance(wass_temp,float)
    assert isinstance(wass_prices,float)


def test_with_same_data()->ContinuousFeatureDrift:
    api_key=os.getenv("MonitorAI_API_KEY")
    client=MonitorAI(api_key)
    ref_df=pd.DataFrame({
                "temperature":[11,14,15,37,55,88,90,41,19,19],
                "prices":[15,18,43,55,67,91,110,21,39,81]
            })


    return ContinuousFeatureDrift(client=client,
                                ref_data=ref_df,
                                analysis_data=ref_df)

def test_same_bhatta_coeff():
    cont_drift=test_with_same_data()
    bhatta_temp=cont_drift.bhatta_coeff(feature="temperature")
    bhatta_prices=cont_drift.bhatta_coeff(feature="prices")


    assert isinstance(bhatta_temp,float)
    assert isinstance(bhatta_prices,float)

    assert bhatta_temp==0
    assert bhatta_prices==0


def test_same_euclidean_dist():
    cont_drift=test_with_same_data()
    euclid_temp=cont_drift.euclidean_dist("temperature")
    euclid_prices=cont_drift.euclidean_dist("prices")

    assert isinstance(euclid_temp,float)
    assert isinstance(euclid_prices,float)

    assert euclid_temp==0
    assert euclid_prices==0

def test_same_cauchy_schwartz_dist():
    cont_drift=feature_drift()
    cs_dist_temp=cont_drift.cauchy_schwartz_dist("temperature")
    cs_dist_prices=cont_drift.cauchy_schwartz_dist("prices")

    assert isinstance(cs_dist_temp,float)
    assert isinstance(cs_dist_prices,float)

    assert cs_dist_temp==0
    assert cs_dist_prices==0


def test_same_hellinger_dist():
    cont_drift=feature_drift()
    hellinger_temp=cont_drift.hellinger_dist("temperature")
    hellinger_prices=cont_drift.hellinger_dist("prices")

    assert isinstance(hellinger_temp,float)
    assert isinstance(hellinger_prices,float)

    assert hellinger_temp==0
    assert hellinger_prices==0

def test_same_js_div():
    cont_drift=feature_drift()
    js_temp=cont_drift.js_divergence("temperature")
    js_prices=cont_drift.js_divergence("prices")

    assert isinstance(js_temp,float)
    assert isinstance(js_prices,float)

    assert js_temp==0
    assert js_prices==0

def test_same_kl_div():
    cont_drift=feature_drift()
    kl_temp=cont_drift.kl_divergence("temperature")
    kl_prices=cont_drift.kl_divergence("prices")

    assert isinstance(kl_temp,float)
    assert isinstance(kl_prices,float)

    assert kl_temp==0
    assert kl_prices==0

def test_same_psi():
    cont_drift=feature_drift()
    psi_temp=cont_drift.psi("temperature")
    psi_prices=cont_drift.psi("prices")

    assert isinstance(psi_temp,float)
    assert isinstance(psi_prices,float)

    assert psi_temp==0
    assert psi_prices==0

def test_same_tvd():
    cont_drift=feature_drift()
    tvd_temp=cont_drift.total_variation_dist("temperature")
    tvd_prices=cont_drift.total_variation_dist("prices")

    assert isinstance(tvd_temp,float)
    assert isinstance(tvd_prices,float)

    assert tvd_temp==0
    assert tvd_prices==0

def test_same_wasserstein():
    cont_drift=feature_drift()
    wass_temp=cont_drift.wasserstein_dist("temperature")
    wass_prices=cont_drift.wasserstein_dist("prices")

    assert isinstance(wass_temp,float)
    assert isinstance(wass_prices,float)

    assert wass_temp==0
    assert wass_prices==0
