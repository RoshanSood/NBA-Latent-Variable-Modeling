# Predicting NBA Game Outcomes Using Latent Team Strengths + Game-Level Features

This project builds a probabilistic model to estimate:

\[
P(\text{Home Team Wins} \mid \text{Team Strengths}, \text{Game Features})
\]

The approach combines **latent team strength parameters**, **observable game-level covariates**, and **maximum-likelihood / MAP estimation** within a logistic regression framework. The model is inspired by Bayesian networks but optimized using gradient-based inference.

The code supports:

- Feature engineering (rest days, point differential)  
- Latent strength learning  
- Logistic model training with a Gaussian prior  
- Validation-based hyperparameter selection  
- Test-set probability predictions

---

## 1. Dataset

We use the NBA games dataset from Kaggle:  
<https://www.kaggle.com/datasets/nathanlauga/nba-games>

Key fields:

- `HOME_TEAM_ID`, `VISITOR_TEAM_ID`  
- `PTS_home`, `PTS_away` (or `HOME_TEAM_POINTS`, `VISITOR_TEAM_POINTS`)  
- `HOME_TEAM_WINS` (binary label)  
- `GAME_DATE_EST`, `SEASON`

By default, the model uses:

- **Train:** seasons 2014–2020  
- **Validation:** season 2021  
- **Test:** season 2022  

If your local copy does not contain 2021–2022, you may want to adjust the year ranges in the code.

---

## 2. Feature Engineering

We construct three game-level covariates:

### 2.1 Rest Days

For each team, we compute the number of days since its previous game:

\[
\text{RestDays}_t(g) = \max\big( (\text{Date}_g - \text{Date}_{g-1}),\; 0 \big)
\]

For each game \(g\):

- `RestDaysHome` = rest days of the home team before game \(g\)  
- `RestDaysAway` = rest days of the away team before game \(g\)

If a team has no previous game (e.g., first game of the season), we assign a default rest value (e.g., 3 days).

---

### 2.2 Point Differential

We compute the home–away point differential:

\[
\text{PointDifferential} = \text{Points}_{\text{Home}} - \text{Points}_{\text{Away}}
\]

In practice:

- If the columns `PTS_home` and `PTS_away` exist:
  \[
  \text{PointDifferential} = \text{PTS\_home} - \text{PTS\_away}
  \]
- Otherwise, if the columns `HOME_TEAM_POINTS` and `VISITOR_TEAM_POINTS` exist:
  \[
  \text{PointDifferential} = \text{HOME\_TEAM\_POINTS} - \text{VISITOR\_TEAM\_POINTS}
  \]

This provides a simple summary of scoring dominance that can be used as a feature.

---

### 2.3 Label: Home Win Indicator

The outcome variable is:

\[
Y =
\begin{cases}
1 & \text{if the home team wins the game} \\
0 & \text{otherwise}
\end{cases}
\]

The dataset column corresponding to this is `HOME_TEAM_WINS`.

---

## 3. Model Specification

The goal is to model the probability that the home team wins a game given:

- The **latent strength** of each team  
- The **observable game-level features** of that specific matchup

Formally, we want:

\[
P(Y = 1 \mid s_h, s_a, X)
\]

where:

- \( s_h \) is the latent strength of the home team  
- \( s_a \) is the latent strength of the away team  
- \( X \) is the feature vector for the game (e.g., rest days, point differential)  
- \( Y \in \{0,1\} \) indicates whether the home team wins

---

### 3.1 Logistic Model

We assume a logistic (sigmoid) link function:

\[
P(Y = 1 \mid s_h, s_a, X) = \sigma(z)
\]

with the **logit**:

\[
z = \mu + (s_h - s_a) + \beta^\top X
\]

where:

- \( \mu \in \mathbb{R} \) is a scalar intercept capturing **global home-court advantage**  
- \( s_h, s_a \in \mathbb{R} \) are the latent strengths of the home and away teams  
- \( \beta \in \mathbb{R}^d \) is the feature weight vector  
- \( X \in \mathbb{R}^d \) is the game feature vector for that matchup  
- \( \sigma(\cdot) \) is the logistic sigmoid:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

In this project:

\[
X = \begin{bmatrix}
\text{RestDaysHome} \\
\text{RestDaysAway} \\
\text{PointDifferential}
\end{bmatrix}
\]

so:

\[
\beta^\top X = \beta_{\text{RH}} \cdot \text{RestDaysHome}
+ \beta_{\text{RA}} \cdot \text{RestDaysAway}
+ \beta_{\text{PD}} \cdot \text{PointDifferential}
\]

The quantity \( s_h - s_a \) captures the relative strength of the two teams: if \( s_h > s_a \), the home team is “stronger” in expectation.

---

## 4. Parameter Estimation

We estimate parameters:

\[
\theta = \{\mu, \beta, s_1, s_2, \dots, s_T\}
\]

where \( T \) is the number of teams. The estimation is done via **Maximum A Posteriori (MAP)**, which is equivalent to **Maximum Likelihood Estimation (MLE)** with an added prior penalty on the team strengths.

### 4.1 Likelihood for a Single Game

For a single game \( i \) with outcome \( Y_i \) and logit \( z_i \):

\[
P(Y_i \mid z_i) =
\begin{cases}
\sigma(z_i) & \text{if } Y_i = 1 \\
1 - \sigma(z_i) & \text{if } Y_i = 0
\end{cases}
\]

The corresponding log-likelihood is:

\[
\ell_i(\theta)
= Y_i \log \sigma(z_i) + (1 - Y_i)\log\big(1 - \sigma(z_i)\big)
\]

### 4.2 Likelihood for All Games

Given \( N \) games in the dataset:

\[
\mathcal{L}(\theta) = \sum_{i=1}^N \ell_i(\theta)
= \sum_{i=1}^N \Big[ Y_i \log \sigma(z_i) + (1 - Y_i)\log\big(1 - \sigma(z_i)\big) \Big]
\]

In practice, we optimize the **negative average log-likelihood**:

\[
\text{NLL}(\theta) = -\frac{1}{N}\mathcal{L}(\theta)
\]

This corresponds to what is typically implemented as binary cross-entropy loss.

---

### 4.3 Gaussian Prior on Team Strengths

To prevent the latent strengths \( s_t \) from becoming arbitrarily large in magnitude and to encode the belief that teams are roughly centered around zero strength, we place a Gaussian prior:

\[
s_t \sim \mathcal{N}(0, \sigma^2)
\]

for each team \( t \). The log-prior for all teams is:

\[
\log P(\{s_t\}) = -\sum_{t=1}^T \frac{s_t^2}{2\sigma^2} + \text{const.}
\]

This leads to an **L2 regularization** term in the loss:

\[
\text{Penalty}(\theta)
= \frac{1}{2\sigma^2}\sum_{t=1}^T s_t^2
\]

We scale this penalty by a hyperparameter \( \lambda_{\text{strength}} \) to control its impact:

\[
\lambda_{\text{strength}} \cdot \frac{1}{2\sigma^2}\sum_{t=1}^T s_t^2
\]

---

### 4.4 Final Loss Function (MAP Objective)

Combining the negative log-likelihood and the prior penalty, the final loss we minimize is:

\[
\mathcal{L}_{\text{loss}}(\theta)
= \text{NLL}(\theta) + \lambda_{\text{strength}} \cdot \frac{1}{2\sigma^2}\sum_{t=1}^T s_t^2
\]

Explicitly:

\[
\mathcal{L}_{\text{loss}}(\theta)
= -\frac{1}{N}\sum_{i=1}^N \Big[ Y_i \log \sigma(z_i) + (1 - Y_i)\log\big(1 - \sigma(z_i)\big) \Big]
+ \lambda_{\text{strength}} \cdot \frac{1}{2\sigma^2}\sum_{t=1}^T s_t^2
\]

This is exactly what is implemented in code using:

- `BCEWithLogitsLoss(reduction="mean")` for the NLL  
- An explicit L2 penalty on the `strengths` parameter

Gradient-based optimization (e.g., SGD) is then used to find the parameters that minimize this loss.

---

## 5. Training Procedure

1. **Split the data:**
   - Train: seasons 2014–2020  
   - Validation: season 2021  
   - Test: season 2022  

2. **Standardize features:**
   - Fit a `StandardScaler` on the training features  
   - Apply the same transformation to validation and test features  

3. **Model training for each candidate learning rate** \( \eta \in \{0.001, 0.003, 0.01, 0.03, 0.1\} \):
   - Initialize model parameters
   - Run for a fixed number of epochs (e.g., 50)
   - At each epoch:
     - Compute logits \( z_i = \mu + (s_{h_i} - s_{a_i}) + \beta^\top X_i \)
     - Compute the loss \( \mathcal{L}_{\text{loss}}(\theta) \)
     - Take a gradient step using SGD

4. **Validation-based learning rate selection:**
   - For each learning rate, compute the **average predictive log-likelihood** on the validation set:
     \[
     \text{PLL}_{\text{val}} = \frac{1}{N_{\text{val}}}\sum_{i=1}^{N_{\text{val}}} \log P(Y_i \mid X_i)
     \]
   - Choose the learning rate that yields the highest validation PLL.

5. **Final evaluation on test data:**
   - Using the selected learning rate and corresponding model, evaluate on the test set:
     - Predictive log-likelihood  
     - Brier score  
     - Accuracy  
   - Also output some of the most confident predicted games with their estimated \( P(\text{home wins}) \).

---

## 6. Evaluation Metrics

We compute three main metrics on the test set.

### 6.1 Predictive Log-Likelihood

For test predictions \( p_i = P(Y_i = 1 \mid X_i) \):

\[
\text{PLL} = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \log P(Y_i \mid X_i)
\]

Higher is better. A value close to \(-\log 2 \approx -0.693\) corresponds to always predicting 0.5 on a balanced dataset.

---

### 6.2 Brier Score

The Brier score is the mean squared error between predicted probabilities and actual binary outcomes:

\[
\text{Brier} = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} (p_i - Y_i)^2
\]

- Lower is better  
- A baseline model that predicts 0.5 for all games would have a Brier score around 0.25 on a balanced dataset.

---

### 6.3 Classification Accuracy

We convert probabilities to hard predictions using a 0.5 threshold:

\[
\hat{Y}_i =
\begin{cases}
1 & \text{if } p_i \ge 0.5 \\
0 & \text{otherwise}
\end{cases}
\]

Accuracy is:

\[
\text{Acc} = \frac{1}{N_{\text{test}}}\sum_{i=1}^{N_{\text{test}}} \mathbf{1}\{\hat{Y}_i = Y_i\}
\]

This measures how often the model correctly predicts the winner outright.

---

## 7. Outputs

When you run the training script, it will:

1. Print validation average predictive log-likelihood for each candidate learning rate  
2. Report the selected learning rate based on validation performance  
3. Print test performance:
   - Test predictive log-likelihood  
   - Test Brier score  
   - Test accuracy  
4. Show the **top 10 most confident test predictions**, including:
   - `GAME_DATE_EST`  
   - `SEASON`  
   - `HOME_TEAM_ID`  
   - `VISITOR_TEAM_ID`  
   - `p_home_win` = estimated \( P(\text{home wins}) \)

---

## 8. Files

- `train.py` – complete training pipeline, model definition, and evaluation  
- `games.csv` – NBA games results (from Kaggle)  
- `teams.csv` – team metadata  
- `README.md` – this documentation

---

## 9. How to Run

### 9.1 Install dependencies

```bash
pip install torch numpy pandas scikit-learn
