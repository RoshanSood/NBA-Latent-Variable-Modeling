import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler

# ----------------------------
# 1) Load and preprocess data
# ----------------------------

games = pd.read_csv("games.csv")     # must contain GAME_DATE_EST, HOME_TEAM_ID, VISITOR_TEAM_ID, HOME_TEAM_WINS, etc.
teams = pd.read_csv("teams.csv")     # must contain TEAM_ID, TEAM_NAME, etc.

# Ensure date type and season
games["GAME_DATE_EST"] = pd.to_datetime(games["GAME_DATE_EST"])
# If your CSV already has SEASON or SEASON_ID, use it. Otherwise derive season by game date.
# Simple example: NBA season starts in October, ends by June. Map by year of season start.
def infer_season(d):
    y = d.year
    # if game is in Jul, Aug, Sep, it belongs to previous season year
    if d.month >= 10:
        return y
    else:
        return y - 1
games["SEASON"] = games["GAME_DATE_EST"].apply(infer_season)

# Map team IDs to a dense index [0..T-1]
team_ids = pd.Index(sorted(pd.unique(pd.concat([games["HOME_TEAM_ID"], games["VISITOR_TEAM_ID"]] , axis=0))))
team_to_idx = {tid: i for i, tid in enumerate(team_ids)}
T = len(team_ids)

games["home_idx"] = games["HOME_TEAM_ID"].map(team_to_idx)
games["away_idx"] = games["VISITOR_TEAM_ID"].map(team_to_idx)

# Outcome label
y = games["HOME_TEAM_WINS"].astype(int)

# ------------------------------------------
# 2) Build causal pregame features (no leak)
# ------------------------------------------
# Rest days per team: days since that team's previous game
games = games.sort_values("GAME_DATE_EST").reset_index(drop=True)

def compute_rest_days(df, team_col, date_col):
    # df has columns team_col and date_col
    df = df[[team_col, date_col]].copy()
    df["prev_date"] = df.groupby(team_col)[date_col].shift(1)
    rest = (df[date_col] - df["prev_date"]).dt.days
    return rest

games["RestDaysHome"] = compute_rest_days(games.assign(team=games["home_idx"]), "team", "GAME_DATE_EST")
games["RestDaysAway"] = compute_rest_days(games.assign(team=games["away_idx"]), "team", "GAME_DATE_EST")

# First game of a season for a team will have NaN rest. Impute with a sensible value, for example 3.
games["RestDaysHome"] = games["RestDaysHome"].fillna(3).clip(lower=0, upper=10)
games["RestDaysAway"] = games["RestDaysAway"].fillna(3).clip(lower=0, upper=10)

# Features X. Choose either:
#   A) keep μ as a free scalar parameter and DO NOT include an intercept in X
# or
#   B) include an intercept in X and drop μ
# I choose A). So no constant column in X.
feature_cols = ["RestDaysHome", "RestDaysAway"]
X = games[feature_cols].astype(float).values

# Scale features using train stats only
train_mask = (games["SEASON"] >= 2014) & (games["SEASON"] <= 2020)
val_mask   = (games["SEASON"] == 2021)
test_mask  = (games["SEASON"] == 2022)

scaler = StandardScaler()
X_train = scaler.fit_transform(X[train_mask])
X_val   = scaler.transform(X[val_mask])
X_test  = scaler.transform(X[test_mask])

y_train = y[train_mask].values
y_val   = y[val_mask].values
y_test  = y[test_mask].values

home_train = games.loc[train_mask, "home_idx"].values
away_train = games.loc[train_mask, "away_idx"].values
home_val   = games.loc[val_mask, "home_idx"].values
away_val   = games.loc[val_mask, "away_idx"].values
home_test  = games.loc[test_mask, "home_idx"].values
away_test  = games.loc[test_mask, "away_idx"].values

# ----------------------------
# 3) Torch model definition
# ----------------------------

class BNStrengthLogit(nn.Module):
    def __init__(self, num_teams, num_features):
        super().__init__()
        # latent strengths per team
        self.strengths = nn.Parameter(torch.zeros(num_teams))
        # linear weights for X
        self.beta = nn.Parameter(torch.zeros(num_features))
        # global bias mu
        self.mu = nn.Parameter(torch.zeros(1))

    def forward(self, home_idx, away_idx, X):
        # home_idx, away_idx: Long tensors
        s_home = self.strengths[home_idx]
        s_away = self.strengths[away_idx]
        lin = self.mu + (s_home - s_away) + (X @ self.beta)
        return lin  # logits

# Utility to compute average predictive log-likelihood
def avg_loglik(logits, y_true):
    # logits are raw; y_true in {0,1}
    # log P(y|logit) = -BCEWithLogits(y, logit)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    losses = bce(logits, y_true)
    return (-losses).mean().item()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BNStrengthLogit(T, X_train.shape[1]).to(device)

# Prepare tensors
def to_tensors(X_np, y_np, h_np, a_np):
    return (torch.tensor(X_np, dtype=torch.float32, device=device),
            torch.tensor(y_np, dtype=torch.float32, device=device),
            torch.tensor(h_np, dtype=torch.long, device=device),
            torch.tensor(a_np, dtype=torch.long, device=device))

Xtr, ytr, htr, atr = to_tensors(X_train, y_train, home_train, away_train)
Xva, yva, hva, ava = to_tensors(X_val,   y_val,   home_val,   away_val)
Xte, yte, hte, ate = to_tensors(X_test,  y_test,  home_test,  away_test)

# ----------------------------
# 4) Train by MLE with L2 on strengths
# ----------------------------
bce = nn.BCEWithLogitsLoss(reduction="mean")
optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=200, line_search_fn="strong_wolfe")

# Strength prior variance sigma^2. Larger means weaker penalty.
sigma2 = 1.0
lambda_strength = 1.0 / (2.0 * sigma2)

def closure():
    optimizer.zero_grad()
    logits = model(htr, atr, Xtr)
    loss = bce(logits, ytr)
    # L2 penalty only on strengths to tie down identifiability
    penalty = lambda_strength * torch.sum(model.strengths**2)
    total = loss + penalty
    total.backward()
    return total

optimizer.step(closure)

# Optionally re-center strengths after LBFGS to sum to zero
with torch.no_grad():
    model.strengths -= model.strengths.mean()

# Validation metrics
with torch.no_grad():
    val_logits = model(hva, ava, Xva)
    test_logits = model(hte, ate, Xte)
    val_avg_loglik = avg_loglik(val_logits, yva)
    test_avg_loglik = avg_loglik(test_logits, yte)

print(f"Validation avg predictive log-likelihood: {val_avg_loglik:.6f}")
print(f"Test avg predictive log-likelihood:       {test_avg_loglik:.6f}")

# Also report Brier and accuracy as sanity checks
def brier_score(logits, y_true):
    p = torch.sigmoid(logits)
    return torch.mean((p - y_true)**2).item()

def accuracy(logits, y_true, thr=0.5):
    p = torch.sigmoid(logits)
    pred = (p >= thr).float()
    return torch.mean((pred == y_true).float()).item()

with torch.no_grad():
    print(f"Test Brier: {brier_score(test_logits, yte):.6f}")
    print(f"Test Acc:   {accuracy(test_logits, yte):.4f}")

# ----------------------------
# 5) Use the model to output probabilities for test games
# ----------------------------
with torch.no_grad():
    p_home_win_test = torch.sigmoid(test_logits).cpu().numpy()

# Example: inspect top 10 most confident predictions
conf_idx = np.argsort(np.abs(p_home_win_test - 0.5))[::-1][:10]
out_df = games.loc[test_mask].iloc[conf_idx, :].copy()
out_df["p_home_win"] = p_home_win_test[conf_idx]
out_df = out_df[["GAME_DATE_EST", "HOME_TEAM_ID", "VISITOR_TEAM_ID", "p_home_win"]]
print(out_df.head(10))