import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler

games = pd.read_csv("games.csv")
teams = pd.read_csv("teams.csv")

games["GAME_DATE_EST"] = pd.to_datetime(games["GAME_DATE_EST"])

#check if data is valid
if "SEASON" not in games.columns:
    season = games["GAME_DATE_EST"].dt.year
    #set year to start in sept
    month = games["GAME_DATE_EST"].dt.month
    season = np.where(month <= 8, season - 1, season)
    games["SEASON"] = season

#gather rest day data
games["RestDaysHome"] = np.nan
games["RestDaysAway"] = np.nan

#count the number of rest day to account for fatigue
games_sorted = games.sort_values("GAME_DATE_EST")

all_team_ids = pd.concat([games_sorted["HOME_TEAM_ID"], games_sorted["VISITOR_TEAM_ID"]]).unique()

default_rest_days = 3  

for team_id in all_team_ids:
    mask_team = (games_sorted["HOME_TEAM_ID"] == team_id) | (games_sorted["VISITOR_TEAM_ID"] == team_id)
    team_games = games_sorted.loc[mask_team].sort_values("GAME_DATE_EST")

    prev_date = None
    for idx, row in team_games.iterrows():
        current_date = row["GAME_DATE_EST"]
        if prev_date is None:
            rest_days = default_rest_days
        else:
            rest_days = (current_date - prev_date).days
            #might not need to see if team is not resting
            rest_days = max(rest_days, 0)

        if row["HOME_TEAM_ID"] == team_id:
            games_sorted.at[idx, "RestDaysHome"] = rest_days
        else:
            games_sorted.at[idx, "RestDaysAway"] = rest_days

        prev_date = current_date

#sort to make data readlbe
games = games_sorted.sort_index()

#account for invalid data
games["RestDaysHome"].fillna(games["RestDaysHome"].median(), inplace=True)
games["RestDaysAway"].fillna(games["RestDaysAway"].median(), inplace=True)

#setup train test valid split
train_mask = (games["SEASON"] >= 2014) & (games["SEASON"] <= 2020)
val_mask   = (games["SEASON"] == 2021)
test_mask  = (games["SEASON"] == 2022)
#latent strength ids mapped onto team ids
team_ids = pd.concat([games["HOME_TEAM_ID"], games["VISITOR_TEAM_ID"]]).unique()
team_ids = np.sort(team_ids)
team_id_to_idx = {tid: i for i, tid in enumerate(team_ids)}
num_teams = len(team_ids)

#feature matrix X with label y
feature_cols = ["RestDaysHome", "RestDaysAway"]

def make_split(mask, scaler=None, fit_scaler=False):
    df = games.loc[mask].copy()
    if df.empty:
        return None, None, None, None, None

    home_idx = df["HOME_TEAM_ID"].map(team_id_to_idx).astype(int).values
    away_idx = df["VISITOR_TEAM_ID"].map(team_id_to_idx).astype(int).values

    X = df[feature_cols].values.astype(np.float32)
    y = df["HOME_TEAM_WINS"].values.astype(np.float32)

    if fit_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    #needed tensors
    home_idx = torch.from_numpy(home_idx).long()
    away_idx = torch.from_numpy(away_idx).long()
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()

    return home_idx, away_idx, X, y, scaler

home_tr, away_tr, X_tr, y_tr, scaler = make_split(train_mask, scaler=None, fit_scaler=True)

home_va, away_va, X_va, y_va, _ = make_split(val_mask, scaler=scaler, fit_scaler=False) if val_mask.any() else (None, None, None, None, None)
home_te, away_te, X_te, y_te, _ = make_split(test_mask, scaler=scaler, fit_scaler=False) if test_mask.any() else (None, None, None, None, None)

num_features = len(feature_cols)


#model

#logit P(Y=1) = mu + (s_home - s_away) + beta^T X
#beta are feature weights
#mu is the bias within home court advantage
#strengths are the strengths per team
class BNStrengthLogit(nn.Module):
    def __init__(self, num_teams: int, num_features: int):
        super().__init__()
        #s_t ~ N(0, sigma^2) via L2 penalty
        self.strengths = nn.Parameter(torch.zeros(num_teams))   
        self.beta = nn.Parameter(torch.zeros(num_features))  
        #home advantage   
        self.mu = nn.Parameter(torch.zeros(1))                  

    def forward(self, home_idx, away_idx, X):
        s_home = self.strengths[home_idx]
        s_away = self.strengths[away_idx]
        z = self.mu + (s_home - s_away) + (X @ self.beta)
        return z  

#training

bce_sum = nn.BCEWithLogitsLoss(reduction="sum")
bce_mean = nn.BCEWithLogitsLoss(reduction="mean")
#log-likelihood: (1/N) * sum_i log P(Y_i | model)
def avg_log_likelihood(logits: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        n = y.numel()
        if n == 0:
            return float("nan")
        #reduction="sum" returns -sum log P(y_i | x_i)
        neg_loglik_sum = bce_sum(logits, y)
        return float(-neg_loglik_sum.item() / n)

# brier score = mean (p - y)^2, where p = P(Y=1)
def brier_score(logits: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        p = torch.sigmoid(logits)
        return float(((p - y) ** 2).mean().item())
    
#classification accuracy
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        p = torch.sigmoid(logits)
        preds = (p >= 0.5).float()
        return float((preds == y).float().mean().item())

def train_one_model(lr: float,
                    num_epochs: int,
                    lambda_strength: float = 0.01,
                    sigma_strength: float = 1.0,
                    verbose: bool = False):
    model = BNStrengthLogit(num_teams=num_teams, num_features=num_features)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    N_train = y_tr.numel()
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(home_tr, away_tr, X_tr)

        #negative log-likelihood 
        nll = bce_mean(logits, y_tr)

        #L2 prior on strengths: sum_t s_t^2 / (2 sigma^2)
        prior_penalty = (model.strengths ** 2).sum() / (2.0 * sigma_strength ** 2 * N_train)

        loss = nll + lambda_strength * prior_penalty
        loss.backward()
        optimizer.step()

        if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
            with torch.no_grad():
                train_avg_ll = avg_log_likelihood(logits, y_tr)
                print(f"[lr={lr:.4g}] Epoch {epoch:3d} | "
                      f"Train avg log-lik: {train_avg_ll:.4f} | "
                      f"NLL: {nll.item():.4f}")

    #validation log-likelihood
    model.eval()
    if home_va is not None and y_va is not None and y_va.numel() > 0:
        with torch.no_grad():
            val_logits = model(home_va, away_va, X_va)
            val_avg_ll = avg_log_likelihood(val_logits, y_va)
    else:
        val_avg_ll = float("nan")

    return model, val_avg_ll


#(mle) learning-rate search
candidate_lrs = [0.001, 0.003, 0.01, 0.03, 0.1]
num_epochs = 150

print("Finding best learning rates with validation predictive log-likelihood")

best_lr = None
best_val_ll = -float("inf")
best_model = None

for lr in candidate_lrs:
    model, val_ll = train_one_model(lr=lr, num_epochs=num_epochs, verbose=False)

    print(f"  lr={lr:.4g} -> val avg log-lik = {val_ll:.4f}")
    if not np.isnan(val_ll) and val_ll > best_val_ll:
        best_val_ll = val_ll
        best_lr = lr
        best_model = model

if best_lr is None:
    print("Warning: validation set is empty ")
    #fall back to a default LR
    best_lr = 0.01
    best_model, _ = train_one_model(lr=best_lr, num_epochs=num_epochs, verbose=False)

print(f"\nSelected learning rate: {best_lr:.4g} (val avg log-lik = {best_val_ll:.4f})")


#evaluation on test


best_model.eval()

if home_te is None or y_te is None or y_te.numel() == 0:
    print("Test set is empty (no SEASON == 2022 rows). "
          "Adjust the year ranges if you want a non-empty test set.")
else:
    with torch.no_grad():
        test_logits = best_model(home_te, away_te, X_te)
        test_avg_ll = avg_log_likelihood(test_logits, y_te)
        test_brier = brier_score(test_logits, y_te)
        test_acc = accuracy(test_logits, y_te)

    print("\n--- Test performance on NBA SEASON 2022 ---")
    print(f"Test avg predictive log-likelihood: {test_avg_ll:.6f}")
    print(f"Test Brier score:                 {test_brier:.6f}")
    print(f"Test accuracy:                    {test_acc:.4f}")

    #show a few of the most confident predictions
    with torch.no_grad():
        p_home = torch.sigmoid(test_logits).cpu().numpy()

    conf_idx = np.argsort(np.abs(p_home - 0.5))[::-1][:10]
    test_games = games.loc[test_mask].iloc[conf_idx, :].copy()
    test_games["p_home_win"] = p_home[conf_idx]

    cols_to_show = ["GAME_DATE_EST", "SEASON", "HOME_TEAM_ID", "VISITOR_TEAM_ID", "p_home_win"]
    print("\nMost confident test predictions (top 10):")
    print(test_games[cols_to_show].reset_index(drop=True))
