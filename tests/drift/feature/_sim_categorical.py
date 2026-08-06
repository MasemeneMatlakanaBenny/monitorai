import pytest
import os
import pandas as pd
from src.monitorai import MonitorAI
from src.monitorai.drift.feature import SimilarityCategoricalFeatureDrift


def categorical_drift()->SimilarityCategoricalFeatureDrift:
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



    return SimilarityCategoricalFeatureDrift(client=client,
                                ref_data=ref_df,
                                analysis_data=ana_df)


def test_braun_coeff():
    cat_drift=categorical_drift()
    braun_coeff_color=cat_drift.braun_coeff(feature="color")
    braun_coeff_social=cat_drift.braun_coeff(feature="social")

    assert isinstance(braun_coeff_color,float)
    assert isinstance(braun_coeff_social,float)

    assert braun_coeff_color>=0
    assert braun_coeff_social>=0


def test_dice_off():
    cat_drift=categorical_drift()
    dice_color=cat_drift.dice_coeff("color")
    dice_social=cat_drift.dice_coeff("social")

    assert isinstance(dice_color,float)
    assert isinstance(dice_social,float)

    assert dice_color>=0
    assert dice_social>=0

def test_jaccard_coeff():
    cat_drift=categorical_drift()
    jaccard_color=cat_drift.jaccard_coeff("color")
    jaccard_social=cat_drift.jaccard_coeff("social")

    assert isinstance(jaccard_color,float)
    assert isinstance(jaccard_social,float)

    assert jaccard_social>=0
    assert jaccard_color>=0

def test_overlap_coeff():
    cat_drift=categorical_drift()
    overlap_color=cat_drift.overlap_coeff("color")
    overlap_social=cat_drift.overall ("social")

    assert isinstance(overlap_color,float)
    assert isinstance(overlap_social,float)

    assert overlap_color>=0
    assert overlap_social>=0


def test_tanimoto_coeff():
    cat_drift=categorical_drift()
    tan_color=cat_drift.tanimoto_coeff("color")
    tan_social=cat_drift.tanimoto_coeff("social")

    assert isinstance(tan_color,float)
    assert isinstance(tan_social,float)

    assert tan_color>=0
    assert tan_social>=0



def test_with_same_data()->SimilarityCategoricalFeatureDrift:
    api_key=os.getenv("MonitorAI_API_KEY")
    client=MonitorAI(api_key)
    ref_df=pd.DataFrame({
            "color":["red","red","blue","blue","red","yellow","yellow","red","blue"],
            "social":["twitter","twitter","facebook","facebook","twitter","insta","insta","insta","insta"]
        })



    return SimilarityCategoricalFeatureDrift(client=client,
                                ref_data=ref_df,
                                analysis_data=ref_df)

def test_same_braun_coeff():
    cat_drift=test_with_same_data()
    braun_coeff_color=cat_drift.braun_coeff(feature="color")
    braun_coeff_social=cat_drift.braun_coeff(feature="social")


    assert isinstance(braun_coeff_color,float)
    assert isinstance(braun_coeff_social,float)

    assert braun_coeff_color==0
    assert braun_coeff_social==0


def test_same_dice_coefff():
    cat_drift=test_with_same_data()
    dice_color=cat_drift.dice_coeff("color")
    dice_social=cat_drift.dice_coeff("social")

    assert isinstance(dice_color,float)
    assert isinstance(dice_social,float)

    assert dice_color==0
    assert dice_social==0

def test_jaccard_coeff():
    cat_drift=categorical_drift()
    jaccard_color=cat_drift.jaccard_coeff("color")
    jaccard_social=cat_drift.jaccard_coeff("social")

    assert isinstance(jaccard_color,float)
    assert isinstance(jaccard_social,float)

    assert jaccard_social==0
    assert jaccard_color==0

def test_overlap_coeff():
    cat_drift=categorical_drift()
    overlap_color=cat_drift.overlap_coeff("color")
    overlap_social=cat_drift.overall ("social")

    assert isinstance(overlap_color,float)
    assert isinstance(overlap_social,float)

    assert overlap_color==0
    assert overlap_social==0


def test_tanimoto_coeff():
    cat_drift=categorical_drift()
    tan_color=cat_drift.tanimoto_coeff("color")
    tan_social=cat_drift.tanimoto_coeff("social")

    assert isinstance(tan_color,float)
    assert isinstance(tan_social,float)

    assert tan_color==0
    assert tan_social==0

