# MonitorAI SDK
The Official Python SDK for Monitoring Machines Lab.
MonitorAI is an observabiltiy framework designed for tabular ML to GenAI offering root cause analysis and decision intelligence 
allowing a linkup between ML and decision making for organizations that rely and use ML for solutions.

## Getting started with monitorai sdk for feature drift :
- Install the package:

```powershell
pip install monitorai
```
```python

import pandas as pd
from monitorai import MonitorAI
from monitorai.drift.feature import CategoricalFeatureDrift

client=MonitorAI(api_key="***")

ref_df=pd.DataFrame({
"color":["red","red","blue","blue","red","yellow","yellow","red","blue"],
"social":["twitter","twitter","facebook","facebook","twitter","insta","insta","insta","insta"]})
    
analysis_df=pd.DataFrame({
"color":["red","red","red","blue","blue","yellow"],
"social":["twitter","twitter","facebook","insta","insta","insta"]
})


## create feature drift variable:
feature_drift=CategoricalFeatureDrift(ref_data=ref_df,analysis_data=analysis_df)

## get the bhattacharrya coefficients
bhatta_coeff_social=feature_drift.bhatta_coeff(feature="social")
bhatta_coeff_color=feature_drift.bhatta_coeff(feature="color")

## cauchy schwartz distance:
cauchy_schwartz_social=feature_drift.cauchy_schwartz_dist(feature="social")
cauchy_schwartz_color=feature_drift.cauchy_schwartz_dist(feature="color")

## euclidean distance:
euclid_social=feature_drift.euclidean_dist(feature="social")
euclid_color=feature_drift.euclidean_dist(feature="color")

## hellinger distance:
hellinger_social=feature_drift.hellinger_dist(feature="social")
hellinger_color=feature_drift.hellinger_dist(feature="color")

## jensen-shannon divergence:
js_social=feature_drift.js_divergence(feature="social")
js_color=feature_drift.js_divergence(feature="social")

## kl-divergence:
kl_social=feature_drift.kl_divergence(feature="social")
kl_color=feature_drift.kl_divergence(feature="color")

##psi:
psi_social=feature_drift.psi(feature="social")
psi_color=feature_drift.psi(feature="color")
```

## Citing monitorai

If you use monitorai in your research, please cite it:

```bibtex
@software{monitorai,
  title = {monitorai: A Root Cause and Decision Intelligence Kit For Machine Learning},
  author = {Monitoring Machines Lab},
  year = {2026},
  url = {https://github.com/MasemeneMatlakanaBenny/monitorai},
  doi = {10.5281/zenodo.19646175},
  license = {MIT},
}
```


