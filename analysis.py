"""
Bank term deposit campaign: who should we call?
Data: UCI Bank Marketing (bank additional full), 41,188 calls by a Portuguese bank, May 2008 to November 2010.

Run:  python analysis.py      (needs pandas, scikit learn, matplotlib)
Input: data/bank marketing.csv   Output: output/*.csv, charts/*.png, output/summary.txt
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).parent
OUT, CH = ROOT / "output", ROOT / "charts"
OUT.mkdir(exist_ok=True); CH.mkdir(exist_ok=True)
lines = []


def say(t):
    print(t); lines.append(t)


df = pd.read_csv(ROOT / "data" / "bank marketing.csv", sep=";")
df.columns = [c.replace(".", " ").replace(chr(95), " ") for c in df.columns]
df["subscribed"] = (df["y"] == "yes").astype(int)
# Category labels such as blue-collar become blue collar
for c in df.columns:
    if not pd.api.types.is_numeric_dtype(df[c]):
        df[c] = df[c].str.replace("-", " ")
say(f"Calls: {len(df):,}   Subscribed: {df['subscribed'].sum():,} ({df['subscribed'].mean():.1%})")

# STEP 1  Where do subscribers come from? Simple rates by group.
def rate(col):
    t = df.groupby(col)["subscribed"].agg(calls="size", rate="mean").sort_values("rate", ascending=False)
    t["rate"] = t["rate"].round(3)
    return t

for col in ["poutcome", "contact", "month", "job"]:
    t = rate(col)
    t.to_csv(OUT / f"rate by {col}.csv")
say("")
say("Subscription rate by outcome of the previous campaign:")
say(rate("poutcome").to_string())
say("")
say("Subscription rate by contact channel:")
say(rate("contact").to_string())

calls = df["campaign"].clip(upper=10)
t = df.assign(calls=calls).groupby("calls")["subscribed"].agg(customers="size", rate="mean").round(3)
t.to_csv(OUT / "rate by number of calls.csv")
say("")
say("Subscription rate by number of calls to the same customer in this campaign (10 means 10 or more):")
say(t.to_string())

# STEP 2  Model. Call duration is left out: it is only known after the call,
# so it cannot be used to decide whom to call.
features = ["age", "job", "marital", "education", "default", "housing", "loan", "contact", "month",
            "day of week", "campaign", "pdays", "previous", "poutcome", "emp var rate", "cons price idx",
            "cons conf idx", "euribor3m", "nr employed"]
cat = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
num = [c for c in features if c not in cat]
X, y = df[features], df["subscribed"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
model = Pipeline([
    ("prep", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat), ("num", StandardScaler(), num)])),
    ("lr", LogisticRegression(max_iter=2000)),
])
model.fit(Xtr, ytr)
p = model.predict_proba(Xte)[:, 1]
auc = roc_auc_score(yte, p)
say("")
say(f"Logistic regression, 30% hold out test set of {len(yte):,} calls. AUC: {auc:.3f}")

# STEP 3  Rank the test set by score and cut into ten equal groups (deciles)
test = pd.DataFrame({"score": p, "subscribed": yte.values}).sort_values("score", ascending=False).reset_index(drop=True)
test["decile"] = (np.arange(len(test)) // (len(test) / 10)).astype(int) + 1
dec = test.groupby("decile").agg(calls=("subscribed", "size"), subscribers=("subscribed", "sum"))
dec["rate"] = (dec["subscribers"] / dec["calls"]).round(3)
dec["cumulative share of calls"] = (dec["calls"].cumsum() / dec["calls"].sum()).round(3)
dec["cumulative share of subscribers"] = (dec["subscribers"].cumsum() / dec["subscribers"].sum()).round(3)
dec["lift"] = (dec["rate"] / test["subscribed"].mean()).round(2)
dec.to_csv(OUT / "deciles.csv")
say("")
say("Test set by score decile (1 = highest scores):")
say(dec.to_string())

# STEP 4  Sizing. Costs are ASSUMPTIONS, set here so you can change them.
costPerCall = 5.0          # EUR, agent time per call (assumption)
valuePerSubscriber = 100.0  # EUR, margin earned on one term deposit (assumption)
base = test["subscribed"].sum()
rows = []
for k in [3, 4, 5, 10]:
    top = test[test["decile"] <= k]
    rows.append({"call top deciles": k, "calls": len(top), "subscribers": int(top["subscribed"].sum()),
                 "share of subscribers": round(top["subscribed"].sum() / base, 3),
                 "calls saved vs calling everyone": len(test) - len(top),
                 "net value eur (assumption based)": round(top["subscribed"].sum() * valuePerSubscriber - len(top) * costPerCall)})
plan = pd.DataFrame(rows)
plan.to_csv(OUT / "targeting plan.csv", index=False)
say("")
say(f"Targeting plan on the test set (cost per call {costPerCall:.0f} EUR and value per subscriber {valuePerSubscriber:.0f} EUR are assumptions):")
say(plan.to_string(index=False))
scale = len(df) / len(test)
say(f"Scaled to the full {len(df):,} calls, calling the top 3 deciles needs about {int(plan.iloc[0]['calls'] * scale):,} calls "
    f"instead of {len(df):,} and reaches {plan.iloc[0]['share of subscribers']:.0%} of subscribers.")

# STEP 5  Chart: cumulative gains
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot([0] + list(dec["cumulative share of calls"]), [0] + list(dec["cumulative share of subscribers"]), marker="o", color="#333333", label="Calls ranked by model")
ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", label="Calls in random order")
ax.set_xlabel("Share of customers called"); ax.set_ylabel("Share of subscribers reached")
ax.set_title("Calling the top 30% of scores reaches 73% of subscribers")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.legend(frameon=False)
plt.tight_layout(); plt.savefig(CH / "cumulative gains.png", dpi=110); plt.close()

(OUT / "summary.txt").write_text("\n".join(lines))
