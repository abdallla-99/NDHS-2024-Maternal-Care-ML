"""
Aligned 2024 Nigeria DHS maternal-care dropout model comparison.
- All outcomes restricted to BIDX == 1.
- ANC M14 code 98 excluded.
- Maternal PNC uses maternal M62/M66, not newborn M70.
- Eight models, five PSU-grouped 80/20 holdouts.
- Class-balance weights calculated on training folds only.
- Both unweighted and DHS-weighted test metrics exported.
"""
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone

DTA_PATH = "NGKR8BFL.dta"
raw = pd.read_stata(DTA_PATH, convert_categoricals=False)
d = pd.DataFrame(index=raw.index)
d["twin_birth"] = (raw["b0"] >= 1).astype(int)
d["birth_order"] = raw["bord"]
d["mothers_education"] = raw["v106"].replace([9,98,99], np.nan)
d["religion"] = raw["v130"].replace([9,98,99], np.nan)
d["prev_terminated_pregnancy"] = raw["v228"].replace([9,98,99], np.nan)
d["region"] = raw["v024"]
d["has_electricity"] = np.where(raw["v119"].eq(1),1,np.where(raw["v119"].eq(0),0,np.nan))
d["male_child"] = (raw["b4"] == 1).astype(int)
b11 = pd.to_numeric(raw["b11"], errors="coerce")
d["birth_interval_cat"] = np.nan
d.loc[b11 < 24, "birth_interval_cat"] = 0
d.loc[b11.between(24,47), "birth_interval_cat"] = 1
d.loc[b11 >= 48, "birth_interval_cat"] = 2
d.loc[raw["bord"].eq(1), "birth_interval_cat"] = 3
d["mothers_age_cat"] = np.where(raw["v012"] < 20,0,np.where(raw["v012"] < 35,1,2))
d["urban"] = (raw["v025"] == 1).astype(int)
d["wealth_cat"] = np.where(raw["v190"] <= 2,0,np.where(raw["v190"] == 3,1,np.where(raw["v190"] >= 4,2,np.nan)))
d["currently_in_union"] = np.where(raw["v501"].isin([1,2]),1,np.where(raw["v501"].isin([0,3,4,5]),0,np.nan))
d["using_contraception"] = np.where(raw["v313"].isin([1,2,3]),1,np.where(raw["v313"].eq(0),0,np.nan))
media = ["v157","v158","v159"]
d["media_exposure"] = np.where(raw[media].gt(0).any(axis=1),1,np.where(raw[media].eq(0).all(axis=1),0,np.nan))
water_yes = [11,12,13,14,21,31,41,51,61,62,71,72]
water_no = [32,42,43]
d["improved_water"] = np.where(raw["v113"].isin(water_yes),1,np.where(raw["v113"].isin(water_no),0,np.nan))
san_yes = [11,12,13,21,22,41]
san_no = [14,23,31,42,43]
d["improved_sanitation"] = np.where(raw["v116"].isin(san_yes),1,np.where(raw["v116"].isin(san_no),0,np.nan))
d["mother_employed"] = np.where(raw["v731"].ge(1),1,np.where(raw["v731"].eq(0),0,np.nan))
clean = [1,2,3,4,5,6]
polluting = [7,8,9,10,11,12,13,14,15,16,17]
d["clean_fuel"] = np.where(raw["v161"].isin(clean),1,np.where(raw["v161"].isin(polluting),0,np.nan))
FEATURES = ["twin_birth","birth_order","mothers_education","religion","prev_terminated_pregnancy","region","has_electricity","male_child","birth_interval_cat","mothers_age_cat","urban","wealth_cat","currently_in_union","using_contraception","media_exposure","improved_water","improved_sanitation","mother_employed","clean_fuel"]
recent = raw["bidx"].eq(1)
d["anc_dropout"] = np.nan
d.loc[recent & raw["m14"].eq(0),"anc_dropout"] = 1
d.loc[recent & raw["m14"].between(1,97),"anc_dropout"] = 0
nonfacility = [11,12,33,34]
facility = [21,22,23,24,25,26,27,31,32,35,36,37,38,41,42,46]
d["facility_dropout"] = np.nan
d.loc[recent & raw["m15"].isin(nonfacility),"facility_dropout"] = 1
d.loc[recent & raw["m15"].isin(facility),"facility_dropout"] = 0
d["pnc_dropout"] = np.nan
pnc_yes = recent & (raw["m62"].eq(1) | raw["m66"].eq(1))
pnc_no = recent & raw["m66"].eq(0) & (raw["m62"].isna() | raw["m62"].eq(0))
d.loc[pnc_yes,"pnc_dropout"] = 0
d.loc[pnc_no & ~pnc_yes,"pnc_dropout"] = 1
d["survey_weight"] = raw["v005"] / 1_000_000
d["survey_psu"] = raw["v021"].astype(str)
categorical = [c for c in FEATURES if c != "birth_order"]
preprocessor = ColumnTransformer([
    ("cat", Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("encoder",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]), categorical),
    ("num", Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]), ["birth_order"])
])
models = {
    "Logistic Regression": LogisticRegression(max_iter=1500,random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=8,min_samples_leaf=10,random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300,max_depth=10,min_samples_leaf=5,n_jobs=-1,random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200,learning_rate=0.05,max_depth=3,random_state=42),
    "XGBoost": XGBClassifier(n_estimators=300,learning_rate=0.05,max_depth=6,eval_metric="logloss",n_jobs=-1,random_state=42),
    "LightGBM": LGBMClassifier(n_estimators=300,learning_rate=0.05,max_depth=6,verbosity=-1,n_jobs=-1,random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=150,learning_rate=0.05,random_state=42),
    "Naive Bayes": GaussianNB()
}
seeds = [21,42,63,84,105]
outcomes = {"ANC":"anc_dropout","Facility delivery":"facility_dropout","Maternal PNC":"pnc_dropout"}

import shap, matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.base import clone

selected = {
    "ANC": ("anc_dropout", GradientBoostingClassifier(n_estimators=200,learning_rate=0.05,max_depth=3,random_state=42)),
    "Facility delivery": ("facility_dropout", LogisticRegression(max_iter=1500,random_state=42)),
    "Maternal PNC": ("pnc_dropout", GradientBoostingClassifier(n_estimators=200,learning_rate=0.05,max_depth=3,random_state=42))
}
rng = np.random.default_rng(2026)
perm_rows, shap_rows = [], []
for label, (outcome, estimator) in selected.items():
    sub = d[d[outcome].notna()].copy()
    X, y = sub[FEATURES], sub[outcome].astype(int)
    groups, sw = sub["survey_psu"], sub["survey_weight"].astype(float)
    tr, te = next(GroupShuffleSplit(n_splits=1,test_size=.20,random_state=42).split(X,y,groups))
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    wte = sw.iloc[te].to_numpy()
    pipe = Pipeline([("preprocessor",clone(preprocessor)),("model",clone(estimator))])
    pipe.fit(Xtr,ytr,model__sample_weight=compute_sample_weight("balanced",ytr))
    baseline = roc_auc_score(yte,pipe.predict_proba(Xte)[:,1],sample_weight=wte)
    for feature in FEATURES:
        drops=[]
        for rep in range(10):
            Xperm=Xte.copy()
            Xperm[feature]=rng.permutation(Xperm[feature].to_numpy())
            score=roc_auc_score(yte,pipe.predict_proba(Xperm)[:,1],sample_weight=wte)
            drops.append(baseline-score)
        perm_rows.append({"Outcome":label,"Model":type(estimator).__name__,"Feature":feature,
            "Mean_AUC_drop":np.mean(drops),"SD_AUC_drop":np.std(drops,ddof=1),"Baseline_weighted_AUC":baseline})
    transformed=pipe.named_steps["preprocessor"].transform(Xte)
    names=pipe.named_steps["preprocessor"].get_feature_names_out()
    sample=transformed[np.linspace(0,len(Xte)-1,min(400,len(Xte)),dtype=int)]
    model=pipe.named_steps["model"]
    if label=="Facility delivery":
        bg=pipe.named_steps["preprocessor"].transform(Xtr.iloc[np.linspace(0,len(Xtr)-1,min(100,len(Xtr)),dtype=int)])
        explainer=shap.LinearExplainer(model,bg)
    else:
        explainer=shap.TreeExplainer(model)
    values=explainer.shap_values(sample)
    if isinstance(values,list): values=values[-1]
    values=np.asarray(values)
    if values.ndim==3: values=values[:,:,1]
    for fname, val in zip(names,np.mean(np.abs(values),axis=0)):
        src=fname.split("__",1)[-1]
        matches=[f for f in FEATURES if src==f or src.startswith(f+"_")]
        src=matches[0] if matches else src
        shap_rows.append({"Outcome":label,"Model":type(estimator).__name__,"Feature":src,"Mean_abs_SHAP":float(val)})
perm=pd.DataFrame(perm_rows)
shapdf=pd.DataFrame(shap_rows).groupby(["Outcome","Model","Feature"],as_index=False)["Mean_abs_SHAP"].sum()
perm.to_csv("final_weighted_permutation_importance.csv",index=False)
shapdf.to_csv("final_shap_importance_aggregated.csv",index=False)
for label in selected:
    for source, metric, xlabel, suffix in [
        (perm,"Mean_AUC_drop","Decrease in DHS-weighted ROC-AUC","permutation_importance"),
        (shapdf,"Mean_abs_SHAP","Mean absolute SHAP value","shap_importance")]:
        tmp=source[source.Outcome.eq(label)].sort_values(metric).tail(12)
        fig,ax=plt.subplots(figsize=(8.2,6.2))
        ax.barh(tmp.Feature.str.replace("_"," "),tmp[metric])
        ax.set_xlabel(xlabel); ax.set_title(f"{label}: {suffix.replace('_',' ')}")
        ax.grid(axis="x",alpha=.25); fig.tight_layout()
        fig.savefig(f"figure_{label.lower().replace(' ','_')}_{suffix}.png",dpi=300,bbox_inches="tight")
        plt.close(fig)
print("Saved weighted permutation importance, aggregated SHAP values, and six figures.")
