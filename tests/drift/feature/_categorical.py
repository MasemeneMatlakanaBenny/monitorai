import pytest
import os
import pandas as pd
from src.monitorai import MonitorAI
from src.monitorai.drift.feature import CategoricalFeatureDrift


def categorical_drift()->CategoricalFeatureDrift:
    api_key=os.getenv("MonitorAI_API_KEY")
    client=MonitorAI(api_key)
    ref_df=pd.DataFrame({
            "color":["red","red","blue","blue","red","yellow","yellow","red","blue"],
            "social":["twitter","twitter","facebook","facebook","twitter","insta","insta","insta","insta"]
        })
    
    
    ana_df=pd.DataFrame({
        "color":["red","red","red","blue","blue","yellow"],
        "social":["twitter","twitter","facebook","insta","insta","insta"]
    })



    return CategoricalFeatureDrift(client=client,
                                ref_data=ref_df,
                                analysis_data=ana_df)


def test_bhatta_coeff():
    cat_drift=categorical_drift()
    bhatta_color=cat_drift.bhatta_coef(feature="color")
    bhatta_social=cat_drift.bhatta_coef(feature="social")


    assert isinstance(bhatta_color,float)
    assert isinstance(bhatta_social,float)


def test_euclidean_dist():
    cat_drift=categorical_drift()
    euclid_col=cat_drift.euclidean_dist("color")
    euclid_soc=cat_drift.euclidean_dist("social")

    assert isinstance(euclid_col,float)
    assert isinstance(euclid_soc,float)

def test_cauchy_schwartz_dist():
    cat_drift=categorical_drift()
    cs_dist_col=cat_drift.cauchy_schwartz_dist("color")
    cs_dist_soc=cat_drift.cauchy_schwartz_dist("social")

    assert isinstance(cs_dist_col,float)
    assert isinstance(cs_dist_soc,float)

def test_hellinger_dist():
    cat_drift=categorical_drift()
    hellinger_col=cat_drift.hellinger_dist("color")
    hellinger_soc=cat_drift.hellinger_dist("social")

    assert isinstance(hellinger_col,float)
    assert isinstance(hellinger_soc,float)

def test_js_div():
    cat_drift=categorical_drift()
    js_col=cat_drift.js_divergence("color")
    js_soc=cat_drift.js_divergence("social")

    assert isinstance(js_col,float)
    assert isinstance(js_soc,float)

def test_kl_div():
    cat_drift=categorical_drift()
    kl_col=cat_drift.kl_divergence("color")
    kl_soc=cat_drift.kl_divergence("social")

    assert isinstance(kl_col,float)
    assert isinstance(kl_soc,float)

def test_psi():
    cat_drift=categorical_drift()
    psi_col=cat_drift.psi("color")
    psi_soc=cat_drift.psi("social")

    assert isinstance(psi_col,float)
    assert isinstance(psi_soc,float)

def test_tvd():
    cat_drift=categorical_drift()
    tvd_col=cat_drift.total_variation_dist("color")
    tvd_soc=cat_drift.total_variation_dist("social")

    assert isinstance(tvd_col,float)
    assert isinstance(tvd_soc,float)

def test_with_same_data()->CategoricalFeatureDrift:
    api_key=os.getenv("MonitorAI_API_KEY")
    client=MonitorAI(api_key)
    ref_df=pd.DataFrame({
            "color":["red","red","blue","blue","red","yellow","yellow","red","blue"],
            "social":["twitter","twitter","facebook","facebook","twitter","insta","insta","insta","insta"]
        })



    return CategoricalFeatureDrift(client=client,
                                ref_data=ref_df,
                                analysis_data=ref_df)

def test_bhatta_same():
    cat_drift=test_with_same_data()
    bhatta_color=cat_drift.bhatta_coef("color")
    bhatta_social=cat_drift.bhatta_coef("social")

    assert isinstance(bhatta_color,float)
    assert isinstance(bhatta_social,float)

    assert bhatta_color==0
    assert bhatta_social==0

def test_cauchy_schwartz_same():
    cat_drift=test_with_same_data()
    cauchy_color=cat_drift.cauchy_schwartz_dist("color")
    cauchy_social=cat_drift.cauchy_schwartz_dist("social")

    assert isinstance(cauchy_color,float)
    assert isinstance(cauchy_social,float)

    assert cauchy_color==0
    assert cauchy_social==0

def test_euclidean_same():
    cat_drift=test_with_same_data()
    euclid_col=cat_drift.euclidean_dist("color")
    euclid_social=cat_drift.euclidean_dist("social")

    assert isinstance(euclid_col,float)
    assert isinstance(euclid_social,float)

    assert euclid_col==0
    assert euclid_social==0


def test_hellinger_same():
    cat_drift=test_with_same_data()
    hellinger_color=cat_drift.hellinger_dist("color")
    hellinger_social=cat_drift.hellinger_dist("social")

    assert isinstance(hellinger_color,float)
    assert isinstance(hellinger_social,float)

    assert hellinger_color==0
    assert hellinger_social==0

def test_js_same():
    cat_drift=test_with_same_data()
    js_color=cat_drift.js_divergence("color")
    js_social=cat_drift.js_divergence("social")

    assert isinstance(js_color,float)
    assert isinstance(js_social,float)

    assert js_social==0
    assert js_color==0


def test_kl_same():
    cat_drift=test_with_same_data()
    kl_color=cat_drift.kl_divergence("color")
    kl_social=cat_drift.kl_divergence("social")

    assert isinstance(kl_color,float)
    assert isinstance(kl_social,float)

    assert kl_social==0
    assert kl_color==0

def test_psi_same():
    cat_drift=test_with_same_data()
    psi_color=cat_drift.psi("color")
    psi_social=cat_drift.psi("social")

    assert isinstance(psi_color,float)
    assert isinstance(psi_social,float)

    assert psi_color==0
    assert psi_social==0

def test_tvd_same():
    cat_drift=test_with_same_data()
    tvd_color=cat_drift.total_variation_dist("color")
    tvd_social=cat_drift.total_variation_dist("social")

    assert isinstance(tvd_color,float)
    assert isinstance(tvd_social,float)

    assert tvd_color==0
    assert tvd_social==0
