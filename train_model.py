import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# load data
data = pd.read_csv("dataset.csv")

# features & target
X = data.drop("fraud", axis=1)
y = data["fraud"]

# model
model = RandomForestClassifier(n_estimators=50, max_depth=5)
model.fit(X, y)

# save model
pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model trained & saved!")