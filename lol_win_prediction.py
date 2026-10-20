"""
League of Legends: predicting blueWins from the first 10 minutes of Diamond ranked games.
Group 44: Anton Andersson, Liam Jörgensen, Rasmus Rolin

Problem type : binary classification (blueWins = 1 if the blue team won, else 0)
Algorithm 1  : Logistic Regression, implemented from scratch with NumPy
Algorithm 2  : Random Forest (scikit-learn)

Run:     python lol_win_prediction.py
Output:  printed results, figures in figures/, all numbers in results/results.json
"""

# =============================================================================
# 1. Import libraries
# =============================================================================
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

SEED = 42  # fixed random seed so every run gives the same result
BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "data" / "high_diamond_ranked_10min.csv"
FIG_DIR = BASE / "figures"
RES_DIR = BASE / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

# Colours for the plots: blue team = blue, red team = orange
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#52514e"
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
                     "axes.axisbelow": True, "legend.frameon": False})

results = {}  # every number we report is collected here and saved as JSON


# =============================================================================
# 2. Load the dataset
# =============================================================================
df = pd.read_csv(DATA_PATH)
print(f"Loaded {df.shape[0]} games with {df.shape[1]} columns")


# =============================================================================
# 3. Explore the data
# =============================================================================
n_missing = int(df.isna().sum().sum())
n_duplicates = int(df.duplicated().sum())
class_counts = df["blueWins"].value_counts().sort_index()
print(f"Missing values: {n_missing}, duplicate rows: {n_duplicates}")
print(f"Class balance: blue lost = {class_counts[0]}, blue won = {class_counts[1]}")

# Many columns are copies or simple formulas of other columns. Check that this is true.
redundancy_checks = {
    "redFirstBlood = 1 - blueFirstBlood": (df.redFirstBlood == 1 - df.blueFirstBlood).all(),
    "redKills = blueDeaths": (df.redKills == df.blueDeaths).all(),
    "redDeaths = blueKills": (df.redDeaths == df.blueKills).all(),
    "redGoldDiff = -blueGoldDiff": (df.redGoldDiff == -df.blueGoldDiff).all(),
    "redExperienceDiff = -blueExperienceDiff": (df.redExperienceDiff == -df.blueExperienceDiff).all(),
    "blueGoldDiff = blueTotalGold - redTotalGold": (df.blueGoldDiff == df.blueTotalGold - df.redTotalGold).all(),
    "blueExperienceDiff = blueTotalExperience - redTotalExperience":
        (df.blueExperienceDiff == df.blueTotalExperience - df.redTotalExperience).all(),
    "blueGoldPerMin = blueTotalGold / 10": np.allclose(df.blueGoldPerMin, df.blueTotalGold / 10),
    "blueCSPerMin = blueTotalMinionsKilled / 10": np.allclose(df.blueCSPerMin, df.blueTotalMinionsKilled / 10),
    "blueEliteMonsters = blueDragons + blueHeralds": (df.blueEliteMonsters == df.blueDragons + df.blueHeralds).all(),
}
for rule, holds in redundancy_checks.items():
    print(f"  {holds}  {rule}")

# Unusual values: a few games have very many wards (a normal game has about 10-30)
n_ward_outliers = int(((df.blueWardsPlaced > 100) | (df.redWardsPlaced > 100)).sum())
print(f"Games where a team placed more than 100 wards: {n_ward_outliers}")

# Correlation between each column and the target
correlation = df.drop(columns=["gameId"]).corr()["blueWins"].drop("blueWins")

results["exploration"] = {
    "rows": int(df.shape[0]), "columns": int(df.shape[1]),
    "missing": n_missing, "duplicates": n_duplicates,
    "blue_lost": int(class_counts[0]), "blue_won": int(class_counts[1]),
    "redundancy_checks": {k: bool(v) for k, v in redundancy_checks.items()},
    "wards_median": float(df.blueWardsPlaced.median()),
    "wards_max": int(max(df.blueWardsPlaced.max(), df.redWardsPlaced.max())),
    "ward_outlier_games": n_ward_outliers,
    "top_correlations": correlation.abs().sort_values(ascending=False).head(8).round(3).to_dict(),
}

# Figure 1: class balance
fig, ax = plt.subplots(figsize=(4.2, 3))
bars = ax.bar(["Blue lost (0)", "Blue won (1)"], class_counts.values, color=[ORANGE, BLUE], width=0.55)
ax.bar_label(bars, padding=3)
ax.set_ylabel("Number of games")
ax.set_title("Class balance of blueWins")
ax.set_ylim(0, class_counts.max() * 1.15)
fig.savefig(FIG_DIR / "01_class_balance.png")
plt.close(fig)

# Figure 2: correlation of each blue-team column with blueWins
blue_corr = correlation[[c for c in correlation.index if c.startswith("blue")]].sort_values()
fig, ax = plt.subplots(figsize=(6, 5))
ax.barh(blue_corr.index, blue_corr.values, color=[BLUE if v > 0 else ORANGE for v in blue_corr.values])
ax.axvline(0, color=GREY, linewidth=0.8)
ax.set_xlabel("Correlation with blueWins")
ax.set_title("How strongly each blue-team stat relates to winning")
fig.savefig(FIG_DIR / "02_correlation_with_target.png")
plt.close(fig)

# Figure 3: gold difference for games blue won vs lost
fig, ax = plt.subplots(figsize=(6, 3.4))
bins = np.linspace(-10000, 10000, 61)
ax.hist(df.loc[df.blueWins == 0, "blueGoldDiff"], bins=bins, alpha=0.6, color=ORANGE, label="Blue lost")
ax.hist(df.loc[df.blueWins == 1, "blueGoldDiff"], bins=bins, alpha=0.6, color=BLUE, label="Blue won")
ax.axvline(0, color=GREY, linewidth=0.8, linestyle="--")
ax.set_xlabel("blueGoldDiff at 10 min (blue gold minus red gold)")
ax.set_ylabel("Number of games")
ax.set_title("Gold difference separates wins from losses, but overlaps")
ax.legend()
fig.savefig(FIG_DIR / "03_gold_diff_hist.png")
plt.close(fig)

# Figure 4: real win rate for groups of games with similar gold difference
gold_groups = pd.cut(df.blueGoldDiff, bins=np.arange(-6000, 6001, 1000))
win_rate = df.groupby(gold_groups, observed=True)["blueWins"].mean()
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.plot([g.mid for g in win_rate.index], win_rate.values, color=BLUE, linewidth=2, marker="o")
ax.axhline(0.5, color=GREY, linewidth=0.8, linestyle="--")
ax.set_xlabel("blueGoldDiff at 10 min")
ax.set_ylabel("Blue win rate")
ax.set_ylim(0, 1)
ax.set_title("Win rate rises smoothly with the gold lead (S-shape)")
fig.savefig(FIG_DIR / "04_winrate_by_gold.png")
plt.close(fig)


# =============================================================================
# 4. Preprocess the data
# =============================================================================
# gameId is only an ID number. The other columns can be calculated exactly from
# columns we keep (see the checks in step 3), so they add no new information.
DROP_COLUMNS = {
    "gameId": "ID number, says nothing about the game",
    "redFirstBlood": "= 1 - blueFirstBlood",
    "redKills": "= blueDeaths",
    "redDeaths": "= blueKills",
    "redGoldDiff": "= -blueGoldDiff",
    "redExperienceDiff": "= -blueExperienceDiff",
    "blueGoldPerMin": "= blueTotalGold / 10",
    "redGoldPerMin": "= redTotalGold / 10",
    "blueCSPerMin": "= blueTotalMinionsKilled / 10",
    "redCSPerMin": "= redTotalMinionsKilled / 10",
    "blueEliteMonsters": "= blueDragons + blueHeralds",
    "redEliteMonsters": "= redDragons + redHeralds",
    "blueTotalGold": "replaced by blueGoldDiff (blue gold - red gold)",
    "redTotalGold": "replaced by blueGoldDiff",
    "blueTotalExperience": "replaced by blueExperienceDiff (blue XP - red XP)",
    "redTotalExperience": "replaced by blueExperienceDiff",
}
df_clean = df.drop(columns=list(DROP_COLUMNS))

# No missing values -> nothing to fill in.
# All columns are already numbers (yes/no columns are 0/1) -> no encoding needed.
# The ward outliers are real games, so we keep them. Scaling (step 6) keeps
# large numbers from dominating.
print(f"After cleaning: {df_clean.shape[1] - 1} features")


# =============================================================================
# 5. Define features and target
# =============================================================================
X = df_clean.drop(columns=["blueWins"])  # the inputs (features)
y = df_clean["blueWins"]                 # what we want to predict (target)
feature_names = list(X.columns)
results["preprocessing"] = {"dropped": DROP_COLUMNS, "features": feature_names}


# =============================================================================
# 6. Split into training and test sets, then scale the features
# =============================================================================
# 80 % of the games for training, 20 % kept aside for the final test.
# stratify=y keeps the same share of wins and losses in both parts.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

# Standardisation: z = (x - mean) / std.
# Mean and std come from the TRAINING set only, so the test set stays unseen.
train_mean = X_train.mean()
train_std = X_train.std()
X_train_scaled = ((X_train - train_mean) / train_std).to_numpy()
X_test_scaled = ((X_test - train_mean) / train_std).to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
print(f"Training games: {len(y_train)}, test games: {len(y_test)}")
results["split"] = {"train": len(y_train), "test": len(y_test)}


# =============================================================================
# 7. Algorithm 1: Logistic Regression from scratch
# =============================================================================
class LogisticRegressionScratch:
    """Logistic regression trained with gradient descent.

    Prediction: p = sigmoid(X @ w + b)   = probability that blue wins
    Loss:       binary cross-entropy (log loss)
    """

    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)  # avoid overflow in np.exp for huge values
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)  # one weight per feature, start at 0
        self.b = 0.0                   # bias (intercept)
        self.loss_history = []

        for _ in range(self.n_iterations):
            # 1) predict a probability for every training game
            p = self.sigmoid(X @ self.w + self.b)

            # 2) measure how wrong we are (cross-entropy loss)
            eps = 1e-12  # avoids log(0)
            loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
            self.loss_history.append(loss)

            # 3) gradients: in which direction should w and b move?
            error = p - y
            grad_w = (X.T @ error) / n_samples
            grad_b = np.mean(error)

            # 4) take a small step in the opposite direction of the gradient
            self.w -= self.learning_rate * grad_w
            self.b -= self.learning_rate * grad_b
        return self

    def predict_proba(self, X):
        return self.sigmoid(X @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# Try different learning rates and keep the one with the lowest training loss.
# (We choose using the training loss only, never the test set.)
loss_curves = {}
for rate in [0.001, 0.01, 0.1, 1.0]:
    model = LogisticRegressionScratch(learning_rate=rate, n_iterations=1000).fit(X_train_scaled, y_train)
    loss_curves[rate] = model.loss_history
    print(f"  learning rate {rate:<6} final training loss = {model.loss_history[-1]:.4f}")

best_rate = min(loss_curves, key=lambda rate: loss_curves[rate][-1])
print(f"Best learning rate: {best_rate}")
log_reg = LogisticRegressionScratch(learning_rate=best_rate, n_iterations=1000).fit(X_train_scaled, y_train)
results["logreg_training"] = {
    "learning_rate": best_rate, "n_iterations": 1000,
    "final_loss": {str(k): float(v[-1]) for k, v in loss_curves.items()},
}


# =============================================================================
# 8. Algorithm 2: Random Forest (scikit-learn)
# =============================================================================
# Choose max_depth (how deep each tree may grow) with 5-fold cross-validation
# on the training set. The test set is not used for this choice.
depths = [1, 2, 3, 4, 6, 8, 10, 15, 20, None]
depth_train_acc, depth_cv_acc = [], []
for depth in depths:
    forest = RandomForestClassifier(n_estimators=200, max_depth=depth, random_state=SEED, n_jobs=-1)
    cv_acc = cross_val_score(forest, X_train_scaled, y_train, cv=5).mean()
    train_acc = forest.fit(X_train_scaled, y_train).score(X_train_scaled, y_train)
    depth_cv_acc.append(cv_acc)
    depth_train_acc.append(train_acc)
    print(f"  max_depth={str(depth):<5} train accuracy = {train_acc:.4f}   CV accuracy = {cv_acc:.4f}")

best_depth = depths[int(np.argmax(depth_cv_acc))]
print(f"Best max_depth: {best_depth}")

rf = RandomForestClassifier(n_estimators=200, max_depth=best_depth, random_state=SEED, n_jobs=-1)
rf.fit(X_train_scaled, y_train)

# A forest with no depth limit (the default), to show overfitting
rf_default = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
rf_default.fit(X_train_scaled, y_train)

results["rf_tuning"] = {"n_estimators": 200, "depths": [str(d) for d in depths],
                        "train_acc": depth_train_acc, "cv_acc": depth_cv_acc,
                        "best_depth": best_depth}


# =============================================================================
# 9. Evaluate both models on the test set
# =============================================================================
def evaluate(model):
    y_pred = model.predict(X_test_scaled)
    return {
        "train_accuracy": float(accuracy_score(y_train, model.predict(X_train_scaled))),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


lr_metrics = evaluate(log_reg)
rf_metrics = evaluate(rf)
rf_default_metrics = evaluate(rf_default)

# Two simple baselines to compare against
most_common_class = int(np.mean(y_train) >= 0.5)
majority_acc = float(accuracy_score(y_test, np.full(len(y_test), most_common_class)))
gold_rule_acc = float(accuracy_score(y_test, (X_test["blueGoldDiff"] > 0).astype(int)))


# =============================================================================
# 10. Compare the results
# =============================================================================
table = pd.DataFrame({
    "Logistic Regression": lr_metrics,
    "Random Forest (tuned)": rf_metrics,
    "Random Forest (no depth limit)": rf_default_metrics,
}).drop(index="confusion_matrix").astype(float).round(4)
print("\n=== Results on the test set ===")
print(table.to_string())
print(f"Baseline 'always guess the most common class': {majority_acc:.4f}")
print(f"Baseline 'the team with more gold wins':       {gold_rule_acc:.4f}")

weights = pd.Series(log_reg.w, index=feature_names).sort_values()
importance = pd.Series(rf.feature_importances_, index=feature_names).sort_values()

results["test"] = {"logreg": lr_metrics, "rf": rf_metrics, "rf_default": rf_default_metrics,
                   "majority_baseline": majority_acc, "gold_rule_baseline": gold_rule_acc}
results["logreg_weights"] = weights.round(4).to_dict()
results["logreg_bias"] = float(log_reg.b)
results["rf_importance"] = importance.round(4).to_dict()


# =============================================================================
# 11. Visualisations of the models
# =============================================================================
# Figure 5: training loss for different learning rates
fig, ax = plt.subplots(figsize=(6, 3.4))
for rate, color in zip(loss_curves, ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]):
    ax.plot(loss_curves[rate], color=color, linewidth=2, label=f"learning rate = {rate}")
ax.set_xlabel("Iteration")
ax.set_ylabel("Training loss")
ax.set_title("Gradient descent: the loss goes down during training")
ax.legend()
fig.savefig(FIG_DIR / "05_loss_curves.png")
plt.close(fig)

# Figure 6: confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
for ax, (name, metrics) in zip(axes, [("Logistic Regression", lr_metrics), ("Random Forest", rf_metrics)]):
    cm = np.array(metrics["confusion_matrix"])
    ax.imshow(cm, cmap="Blues", vmin=0)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Predicted: lost", "Predicted: won"])
    ax.set_yticks([0, 1], ["Actual: lost", "Actual: won"])
    ax.set_title(name)
    ax.grid(False)
fig.tight_layout()
fig.savefig(FIG_DIR / "06_confusion_matrices.png")
plt.close(fig)

# Figure 7: which features matter most in each model
fig, axes = plt.subplots(1, 2, figsize=(8.4, 6))
axes[0].barh(weights.index, weights.values, color=[BLUE if v > 0 else ORANGE for v in weights.values])
axes[0].axvline(0, color=GREY, linewidth=0.8)
axes[0].set_title("Logistic Regression weights")
axes[0].set_xlabel("Weight  (+ helps blue win, - helps red win)")
axes[1].barh(importance.index, importance.values, color=BLUE)
axes[1].set_title("Random Forest feature importance")
axes[1].set_xlabel("Importance")
fig.tight_layout()
fig.savefig(FIG_DIR / "07_feature_importance.png")
plt.close(fig)

# Figure 8: overfitting - training vs cross-validation accuracy for each depth
depth_labels = [str(d) for d in depths]
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot(depth_labels, depth_train_acc, color=ORANGE, linewidth=2, marker="o", label="Training accuracy")
ax.plot(depth_labels, depth_cv_acc, color=BLUE, linewidth=2, marker="o", label="Cross-validation accuracy")
ax.set_xlabel("max_depth of each tree (None = no limit)")
ax.set_ylabel("Accuracy")
ax.set_title("Deeper trees memorise the training data (overfitting)")
ax.legend()
fig.savefig(FIG_DIR / "08_rf_depth_overfitting.png")
plt.close(fig)

with open(RES_DIR / "results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\nFigures saved in {FIG_DIR}")
print(f"Numbers saved in {RES_DIR / 'results.json'}")
