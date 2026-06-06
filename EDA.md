# Exploratory Data Analysis (EDA)

## 1. Competition Overview

The objective of this competition is to predict whether a player will score in a given football match (`scored_flag`).

**Evaluation Metric:** Average Precision (AP)

### Dataset Shape

| Dataset | Rows      | Columns |
| ------- | --------- | ------- |
| Train   | 1,319,813 | 64      |
| Test    | 565,635   | 63      |

---

## 2. Target Variable Analysis

The target variable is highly imbalanced.

| Class | Percentage |
| ----- | ---------- |
| False | 91.46%     |
| True  | 8.54%      |

### Insight

Only a small percentage of player appearances result in a goal. Therefore, Average Precision (AP) was chosen as the evaluation metric instead of simple accuracy.

---

## 3. Data Type Analysis

The dataset contains a mixture of numerical, categorical and boolean features.

| Type    | Count |
| ------- | ----- |
| Object  | 20    |
| Float   | 20    |
| Integer | 10    |
| Boolean | 14    |

Key information available includes:

* Player attributes
* Market value information
* Match context
* Team statistics
* Position information
* Understat analytics

---

## 4. Missing Value Analysis

Significant missing values were observed in Understat-related features.

| Feature   | Missing % |
| --------- | --------- |
| avg_xG    | ~48%      |
| avg_xA    | ~48%      |
| avg_shots | ~48%      |
| avg_npxG  | ~48%      |

To understand whether these missing values carried information, scoring rates were compared.

| Understat Available | Goal Rate |
| ------------------- | --------- |
| False               | 7.70%     |
| True                | 9.31%     |

### Insight

Players with available Understat data tend to score more frequently. Therefore, the availability of analytics data itself contains useful information.

---

## 5. Position Analysis

Scoring probability varies significantly across positions.

| Position   | Goal Rate |
| ---------- | --------- |
| Attack     | 17.27%    |
| Midfield   | 7.75%     |
| Defender   | 3.72%     |
| Goalkeeper | 0.016%    |

### Insight

Attacking players score substantially more often than defenders and goalkeepers, making position one of the strongest predictors.

---

## 6. Sub-Position Analysis

A deeper analysis was performed using player sub-positions.

| Sub Position       | Goal Rate |
| ------------------ | --------- |
| Centre-Forward     | 21.01%    |
| Second Striker     | 19.82%    |
| Left Winger        | 13.19%    |
| Right Winger       | 12.99%    |
| Attacking Midfield | 12.91%    |
| Goalkeeper         | 0.016%    |

### Insight

Goal-scoring probability differs significantly even within the same broad position category. Centre-forwards and second strikers exhibit the highest scoring rates.

---

## 7. Player Overlap Analysis

A comparison of player identities between train and test sets revealed significant overlap.

| Metric         | Count  |
| -------------- | ------ |
| Train Players  | 27,282 |
| Test Players   | 24,850 |
| Common Players | 24,012 |

### Insight

Most players appearing in the test set were already present in the training set, suggesting player-related information could be valuable.

---

## 8. Goal Feature Analysis

Team goals showed a strong relationship with player scoring probability.

Observations:

* Higher home team goals corresponded to higher player scoring rates.
* Higher away team goals corresponded to higher player scoring rates.
* Larger goal differences were associated with increased scoring probability.

### Insight

Match outcomes and team offensive performance contain important predictive information.

---

## 9. Feature Engineering

Several football-specific features were created.

### Attacking Index

```python
attacking_index = avg_xG + avg_npxG + avg_shots
```

This feature captures a player's attacking threat.

### Creative Index

```python
creative_index = avg_xA + avg_key_passes
```

This feature captures a player's creative contribution.

### Result

Model performance improved from:

```text
AP = 0.3996
```

to

```text
AP = 0.4252
```

making this the most successful feature engineering step.

---

## 10. Modeling Approach

To avoid player leakage, GroupKFold validation was used.

```python
groups = train['name_y']
```

This ensures that the same player does not appear in both training and validation folds.

### Models Tested

| Model    | AP Score |
| -------- | -------- |
| LightGBM | 0.42519  |
| CatBoost | 0.42779  |
| Ensemble | 0.42802  |

### Best Model

CatBoost with engineered football features achieved the best standalone validation performance.

---

## Conclusion

The analysis demonstrated that:

* Position and sub-position strongly influence scoring probability.
* Understat analytics provide valuable predictive information.
* Team goal statistics are highly informative.
* Football-specific feature engineering significantly improves model performance.
* CatBoost outperformed LightGBM on the validation set.

The final solution combined football domain knowledge with machine learning techniques to improve predictive performance on the goal-scoring task.
