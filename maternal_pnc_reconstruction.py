# RECONSTRUCT MATERNAL PNC OUTCOME FROM ORIGINAL 2024 NDHS KR FILE
# M62 = mother's health checked before discharge
# M66 = mother's health checked after discharge / after home delivery
# M70 is newborn health check and is NOT used for maternal PNC.

import numpy as np
import pandas as pd

dhs_raw = pd.read_stata(
    "/content/NGKR8BDT/NGKR8BFL.dta",
    convert_categoricals=False
)

# Maternity-care variables are aligned to the most recent birth.
recent_birth = dhs_raw["bidx"].eq(1)

df["pnc_dropout"] = np.nan

maternal_check_yes = (
    recent_birth &
    (dhs_raw["m62"].eq(1) | dhs_raw["m66"].eq(1))
)

# For home births M62 is structurally missing; M66 is the applicable
# after-delivery check. For facility births, M62 and M66 are both observed.
maternal_check_no = (
    recent_birth &
    dhs_raw["m66"].eq(0) &
    (dhs_raw["m62"].isna() | dhs_raw["m62"].eq(0))
)

# Outcome coding: 1 = no maternal postnatal check (dropout);
# 0 = maternal postnatal check received.
df.loc[maternal_check_yes, "pnc_dropout"] = 0
df.loc[maternal_check_no & ~maternal_check_yes, "pnc_dropout"] = 1

# Never use M70 for this maternal outcome.
assert "pnc_dropout" in df.columns
assert "pnc_dropout" not in FEATURES

print("Maternal PNC outcome audit")
print(df["pnc_dropout"].value_counts(dropna=False).sort_index())
print("Valid N:", int(df["pnc_dropout"].notna().sum()))
print("Dropout N:", int((df["pnc_dropout"] == 1).sum()))
print("Received PNC N:", int((df["pnc_dropout"] == 0).sum()))
