# Customer Churn Prediction

This project predicts customer churn by combining billing data with customer-care calls, emails, and renewal-call interactions. The workflow is notebook-driven and builds a modelling dataset from raw operational tables, tests feature hypotheses, engineers business-focused features, and compares multiple machine learning models.

## Project Goal

The business objective is to identify customers who are likely to churn after the renewal period so retention teams can intervene earlier and more effectively.

The modelling task focuses on the binary classification problem:

- `Won` = retained customer
- `Churned` = lost customer

Rows with `Prospect_Outcome = Open` are retained in the processed data but excluded from the final binary modelling notebooks.

## Project Structure

```text
data/
  raw/
  processed/
notebooks/
  cleaning/
  Merge/
  EDA/
  hypothesis/
  features/
  Model-training/
deliverables/
src/
README.md
```

Key files:

- [notebooks/cleaning/01_billings.ipynb](C:/Users/madha/OneDrive/Desktop/DS/Churn-Prediction-JMAN/notebooks/cleaning/01_billings.ipynb)
- [notebooks/cleaning/02_cc_calls.ipynb](C:/Users/madha/OneDrive/Desktop/DS/Churn-Prediction-JMAN/notebooks/cleaning/02_cc_calls.ipynb)
- [notebooks/cleaning/03_emails.ipynb](C:/Users/madha/OneDrive/Desktop/DS/Churn-Prediction-JMAN/notebooks/cleaning/03_emails.ipynb)
- [notebooks/cleaning/04_renewal_calls.ipynb](C:/Users/madha/OneDrive/Desktop/DS/Churn-Prediction-JMAN/notebooks/cleaning/04_renewal_calls.ipynb)
- [notebooks/Merge/01_merging.ipynb](C:/Users/madha/OneDrive/Desktop/DS/Churn-Prediction-JMAN/notebooks/Merge/01_merging.ipynb)
- [notebooks/EDA/01_master_eda.ipynb](C:/Users/madha/OneDrive/Desktop/DS/Churn-Prediction-JMAN/notebooks/EDA/01_master_eda.ipynb)
- [notebooks/features/01_feature_engg.ipynb](C:/Users/madha/OneDrive/Desktop/DS/Churn-Prediction-JMAN/notebooks/features/01_feature_engg.ipynb)
- [notebooks/hypothesis/04_hypothesis_merged.ipynb](C:/Users/madha/OneDrive/Desktop/DS/Churn-Prediction-JMAN/notebooks/hypothesis/04_hypothesis_merged.ipynb)
- [notebooks/Model-training](C:/Users/madha/OneDrive/Desktop/DS/Churn-Prediction-JMAN/notebooks/Model-training)

## Data Overview

### Raw tables

| Dataset | Rows | Columns | Purpose |
|---|---:|---:|---|
| `billings.csv` | 122,082 | 59 | Master renewal and subscription table |
| `cc_calls.csv` | 32,882 | 33 | Pre-renewal customer-care interactions |
| `emails.csv` | 123,389 | 27 | Pre-renewal email and CRM interaction history |
| `renewal_calls.csv` | 186,534 | 41 | Renewal-call transcripts and follow-up interactions |

### Processed outputs

| Dataset | Rows | Columns | Description |
|---|---:|---:|---|
| `cleaned_billings.csv` | 122,082 | 53 | Cleaned billing table |
| `cleaned_cc_calls.csv` | 32,882 | 33 | Cleaned customer-care calls |
| `cleaned_emails.csv` | 123,389 | 27 | Cleaned emails/CRM data |
| `cleaned_renewal_calls.csv` | 127,307 | 45 | Cleaned renewal-call interactions |
| `master_churn_dataset.csv` | 122,082 | 136 | Merged customer-year master table |
| `model_ready_dataset.csv` | 122,082 | 61 | Final engineered dataset for modelling |

### Target distribution in the final model-ready dataset

- `Won`: 101,226
- `Churned`: 12,668
- `Open`: 8,188

Binary modelling therefore uses a churn rate of roughly 11% among the `Won`/`Churned` subset.

## End-to-End Pipeline

The project follows this sequence:

1. Clean each raw source table independently.
2. Aggregate interactions to customer-year level.
3. Merge everything onto the billing table.
4. Perform EDA and hypothesis testing.
5. Engineer composite and business-derived features.
6. Remove leaky, redundant, and high-cardinality features.
7. Train and compare multiple classification models.

## Data Cleaning

### 1. Billings table cleaning

Notebook: [01_billings.ipynb](C:/Users/madha/OneDrive/Desktop/DS/Churn-Prediction-JMAN/notebooks/cleaning/01_billings.ipynb)

Main cleaning steps:

- Dropped fully empty columns and `Unnamed` columns.
- Dropped highly sparse columns:
  - `Connection_Net`
  - `Connection_Qty`
  - `Starting_Connection_Net`
  - `Starting_Connection_Qty`
  - `Discount_Amount`
- Parsed date fields:
  - `Renewal_Month`
  - `Proforma_Date`
  - `Registration_Date`
  - `Prospect_Renewal_Date`
  - `Closed_Date`
  - `DateTime_Out`
  - `Last_Renewal`
- Standardized boolean-like fields:
  - `Current_Auto_Renewal_Flag`: `y/n` to `Yes/No`
  - `Current_World_Pay_Token`: `y/n` to `Yes/No`
  - `Proforma_Auto_Renewal`: `True/False` to `Yes/No`
  - `Proforma_World_Pay_Token`: `True/False` to `Yes/No`
- Filled missing categoricals with `Unknown` where appropriate:
  - `Band`, `Proforma_Account_Stage`, `Proforma_Audit_Status`, `Proforma_Membership_Status`, `Connection_Group`, `Tenure_Group`, `Anchor_Group`, `Payment_Method`, `Current_Anchor_List`, `Last_Band`
- Normalized `Proforma_Audit_Status` values such as lowercase `vetting`.
- Filled numerics with either `0` or median depending on business meaning.

### 2. Customer-care calls cleaning

Notebook: [02_cc_calls.ipynb](C:/Users/madha/OneDrive/Desktop/DS/Churn-Prediction-JMAN/notebooks/cleaning/02_cc_calls.ipynb)

Main cleaning steps:

- Dropped fully empty and `Unnamed` columns.
- Recovered missing `Co_Ref` values through `Contact_ID` lookup.
- Filled remaining missing `Co_Ref` with `UNKNOWN_<Contact_ID>`.
- Standardized grouped fields:
  - care-package columns to `Not Discussed`
  - sentiment columns to `Neutral`
  - technical issue columns to `No`
- Cleaned noisy Yes/No-like flags by mapping stray text or numeric entries into standard labels.
- Standardized `cc_issues_within_questionnaire` using pattern-based logic so variants of “not applicable” and “not mentioned” become `No`, while actual issue descriptions become `Yes`.
- Cleaned `cc_call_initiated_by` into `Customer`, `Agent`, or `Not Discussed`.
- Normalized `cc_care_package` into a small controlled set:
  - `Standard`
  - `Premier`
  - `Express`
  - `Assisted`
  - `Not Discussed`
- Standardized `cc_contractor_sentiment` to:
  - `Dissatisfied`
  - `Neutral`
  - `Satisfied`
  - `Not Discussed`
- Converted sentiment score columns to numeric and filled with median.
- Parsed `Call_Date`

### 3. Emails / CRM cleaning

Notebook: [03_emails.ipynb](C:/Users/madha/OneDrive/Desktop/DS/Churn-Prediction-JMAN/notebooks/cleaning/03_emails.ipynb)

Main cleaning steps:

- Dropped fully empty and `Unnamed` columns.
- Filled missing categorical and flag fields with `Not Discussed`.
- Standardized CRM Yes/No flags and treated invalid free text as signal-bearing `Yes`.
- Standardized `crm_contractor_sentiment` to:
  - `Neutral`
  - `Satisfied`
  - `Dissatisfied`
  - `Not Discussed`
- Converted `crm_contractor_sentiment_score` to numeric and filled with median.
- Normalized `crm_membership_level` into consistent tiers such as `Gold`, `Silver`, `Bronze`, `Accredited`, `Members Only`, `Standard`, `Premier`, and `Express`.
- Converted `crm_auto_renewal_status` into numeric indicator values.
- Cleaned `crm_agent_chase_count` into numeric counts.

### 4. Renewal-calls cleaning

Notebook: [04_renewal_calls.ipynb](C:/Users/madha/OneDrive/Desktop/DS/Churn-Prediction-JMAN/notebooks/cleaning/04_renewal_calls.ipynb)

Main cleaning steps:

- Dropped fully empty and `Unnamed` columns.
- Filled high-null categorical groups such as:
  - `Churn_Category`
  - `Complaint_Category`
  - `Customer_Reaction_Category`
  - `Agent_Renewal_Pitch_Category`
  - `Customer_Renewal_Response_Category`
  - `Mentioned_Competitors`
  - `Justification_Category`
  - `Reason_For_Renewal_Category`
- Extracted and engineered:
  - `discount_reason`
  - `asked_for_discount`
  - `has_desired_to_cancel`
  - `cancel_reason`
  - `Has_Complaint`
  - `Has_Negative_Reaction`
- Standardized many noisy binary and categorical transcript-derived columns to `Yes`, `No`, `Not Mentioned`, or `Not Applicable`.
- Cleaned `Call_Direction`, `Membership_Renewal_Decision`, and related response categories.
- Dropped rows with missing `Co_Ref`.
- Converted `Call_Date` to datetime and fixed `Call_ID`.
- Collapsed fragmented multiple transcript rows per call/customer into grouped call-level records.

## Merge Logic

Notebook: [01_merging.ipynb](C:/Users/madha/OneDrive/Desktop/DS/Churn-Prediction-JMAN/notebooks/Merge/01_merging.ipynb)

Billings is used as the master table keyed by `(Co_Ref, Renewal_Year)`.

### Email aggregation

Emails are treated as pre-renewal by nature and aggregated by `(Co_Ref, year)` using:

- counts such as `em_email_count`
- proportions of `Yes` for CRM flags
- sentiment score mean and max
- agent chase total and max
- mode features such as:
  - `em_sentiment_mode`
  - `em_membership_level_mode`

### Customer-care calls aggregation

CC calls are first filtered to `Call_Date < Renewal_Month` to keep only pre-renewal signals, then aggregated by `(Co_Ref, Call_Year)` using:

- `cc_call_count`
- proportions of issue/concern/pricing flags
- average sentiment scores
- `cc_sentiment_mode`

### Renewal-calls aggregation

Renewal calls are filtered to `Call_Date > Prospect_Renewal_Date` to retain post-renewal follow-up behavior, then aggregated by `(Co_Ref, Call_Year)` using:

- `ren_call_count`
- binary flag means
- friction score mean/max
- call length and max call number
- mode values for many renewal-call categories

### Post-merge null handling

After left joins onto billings:

- count columns were filled with `0`
- interaction numeric aggregates were filled with `0`
- mode columns were filled with `No Interaction`
- date-derived timeline feature was created:
  - `Days_To_Close_Post_Renewal = Closed_Date - Prospect_Renewal_Date`

Additional dropped date columns after merge:

- `Last_Renewal`
- `Registration_Date`
- `Proforma_Date`

## EDA and Feature Selection Logic

Notebooks:

- [01_master_eda.ipynb](C:/Users/madha/OneDrive/Desktop/DS/Churn-Prediction-JMAN/notebooks/EDA/01_master_eda.ipynb)
- [04_hypothesis_merged.ipynb](C:/Users/madha/OneDrive/Desktop/DS/Churn-Prediction-JMAN/notebooks/hypothesis/04_hypothesis_merged.ipynb)

Feature selection was not done by a single automated selector. Instead, the project used a layered approach:

1. EDA on the merged master dataset
2. Correlation analysis for numeric features
3. Inter-correlation and multicollinearity review
4. Chi-square tests on categorical features
5. Welch’s t-test or Mann-Whitney U on numeric features
6. Business interpretation of renewal, pricing, dissatisfaction, and engagement signals

### Important EDA findings

The strongest predictive signals came from:

- billing and renewal scores
- price and payment behavior
- direct churn-risk signals in emails
- renewal friction features
- selected categorical status and membership features

The EDA highlighted examples such as:

- `Total_Renewal_Score_New` as the strongest overall predictor
- `Total_Net_Paid` and other billing signals as strong negative correlates of churn
- `em_crm_contractor_suggested_leave` as a major churn-risk flag
- post-renewal friction and repeated renewal calls as clear risk indicators

### Hypotheses tested

The merged hypothesis notebook grouped tests into these areas:

- `H1`: Billing and score features
- `H2`: Interaction volume
- `H3`: Sentiment and dissatisfaction
- `H4`: Churn-risk signals and specific issues
- `H5`: Engagement and accreditation health
- `H6`: Renewal friction and complaints
- `H7`: Categorical mode variables via chi-square

### Hypothesis testing conclusions

The key conclusions recorded in the notebook were:

- Billing and score features are highly significant.
- Price-change derived features are highly significant.
- Email-based churn-risk signals are among the strongest predictors.
- Sentiment and dissatisfaction indices are significant.
- Renewal friction and complaint features are significant.
- Many categorical mode variables are statistically associated with churn.

In short, the final feature set was chosen because it was both statistically meaningful and business-interpretable.

## Feature Engineering

Notebook: [01_feature_engg.ipynb](C:/Users/madha/OneDrive/Desktop/DS/Churn-Prediction-JMAN/notebooks/features/01_feature_engg.ipynb)

The feature engineering notebook starts from the 136-column master dataset and reduces it to the 61-column model-ready dataset by removing leakage, collapsing redundancy, and creating derived business signals.

### Step 1: Remove identifiers, date strings, and obvious leakage/redundancy

Dropped:

- `Co_Ref`
- `Renewal_Month`
- `Prospect_Renewal_Date`
- `Closed_Date`
- `DateTime_Out`
- `Current_Anchor_List`
- `Prospect_Status`
- `ren_competitor_benefits_mentioned`
- `Anchor_Group`

Reasons:

- identifiers are not predictive
- raw date strings were not used directly
- `Current_Anchor_List` was high-cardinality free text
- `Prospect_Status` was too granular and partially redundant
- `ren_competitor_benefits_mentioned` was effectively constant
- `Anchor_Group` duplicated `Connection_Group`

### Step 2: Collapse redundant price features

Created:

- `price_change_abs`
- `price_change_pct`
- `net_paid_vs_last`

Dropped:

- `Starting_Net`
- `Starting_Vat`
- `Starting_Gross`
- `Starting_Membership_Net`
- `Starting_Package_Net`
- `Starting_PQQ_Net`
- `Gross`
- `Membership_Net`
- `Package_Net`
- `PQQNet`
- `Amount`
- `Total_Amount`
- `Last_Years_Price`
- `Last_Total_Net_Paid`

Reason:

- many billing amount fields were highly correlated and redundant
- the project kept `Total_Net_Paid` as the main price signal and created more interpretable change features

### Step 3: Collapse redundant connection fields

Created:

- `connection_change`

Dropped:

- `Current_Anchorings`
- `Proforma_Approved_Lists`
- `Last_Connections`

Reason:

- these fields were highly or perfectly correlated with `#_of_Connection`

### Step 4: Aggregate email CRM features into composites

Created:

- `em_accreditation_health`
- `em_churn_risk_signals`
- `em_dissatisfaction_index`
- `em_engagement_signals`

Kept as standalone:

- `em_crm_contractor_suggested_leave`

Dropped the underlying granular component flags after aggregation.

### Step 5: Aggregate customer-care features into composites

Created:

- `cc_dissatisfaction_index`
- `cc_platform_issues_index`
- `cc_pricing_index`
- `cc_engagement_index`
- `cc_sentiment_score_avg`

Dropped the individual underlying CC flags and individual sentiment score columns used to build these composites.

### Step 6: Aggregate renewal-call features

Created:

- `ren_complaint_index`
- `ren_price_sensitivity`
- `ren_competitor_threat`

Dropped:

- complaint component columns after aggregation
- pricing-related component columns after aggregation
- competitor-related component columns after aggregation
- `ren_friction_score_max`

Reason:

- to reduce redundancy and preserve the main behavioral signal in composite form

### Step 7: Additional renewal-call friction feature

Created:

- `ren_Call_Friction_Score`

This feature adds extra weight for direct cancellation and switching intent.

### Step 8: Collapse billing score redundancy

Dropped:

- `Status_Scores`
- `Last_Band`

Reason:

- `Status_Scores` was near-duplicate of `Total_Renewal_Score_New`
- `Last_Band` overlapped with current `Band`

### Step 9: Simplify high-cardinality renewal categorical modes

Created:

- `ren_has_churn_reason`

Dropped:

- `ren_churn_category_mode`
- `ren_complaint_category_mode`
- `ren_customer_reaction_category_mode`
- `ren_agent_renewal_pitch_category_mode`
- `ren_customer_renewal_response_category_mode`
- `ren_agent_response_category_mode`
- `ren_justification_category_mode`
- `ren_reason_for_renewal_category_mode`
- `ren_agent_response_to_cancel_category_mode`
- `ren_argument_that_convinced_customer_to_stay_category_mode`

Reason:

- these fields had high cardinality with limited additional signal relative to simpler summarizations

### Step 10: Derived business features

Created:

- `has_auto_renewal`
- `has_worldpay_token`
- `total_interaction_count`

Dropped after derivation:

- `Proforma_Auto_Renewal`
- `Proforma_World_Pay_Token`
- `Current_Auto_Renewal_Flag`
- `Current_World_Pay_Token`

### Final feature inventory

The final model-ready dataset contains 61 columns including the target. Key model features include:

- billing and score features such as `Total_Renewal_Score_New`, `Auto_Renewal_Score`, `Anchoring_Score`, `Tenure_Scores`, `Total_Net_Paid`
- channel interaction counts such as `em_email_count`, `cc_call_count`, `ren_call_count`
- email composites and sentiment features
- CC call composites and sentiment features
- renewal complaint, friction, and competitor features
- payment and renewal derived features such as `has_auto_renewal`, `has_worldpay_token`, and `total_interaction_count`

## Leakage Handling in Model Training

The model notebooks further drop a small set of leaky or post-outcome features before training. Across the notebooks, the main leaky exclusions are:

- `Total_Net_Paid`
- `price_change_pct`
- `price_change_abs`
- `net_paid_vs_last`
- `Payment_Method`
- `Payment_Timeframe`
- `Days_To_Close_Post_Renewal`
- `Total_Renewal_Score_New`
- `Renewal_Year`

These were excluded from final training because they either reveal too much downstream outcome information, are highly target-adjacent, or are not intended for real-time prediction.

## Modelling Approach

The project trained and evaluated ten model families:

1. AdaBoost
2. Gradient Boosting
3. Logistic Regression
4. LightGBM
5. XGBoost
6. Naive Bayes
7. SVM
8. KNN
9. Decision Tree
10. Random Forest

Most notebooks use:

- train/test split with stratification
- separate baseline and tuned runs
- F1 as the main tuning metric
- class-imbalance handling such as class weighting, SMOTE, or SMOTENC depending on the model

## Model Results

Source: [model_comparison_summary.md](C:/Users/madha/OneDrive/Desktop/DS/Churn-Prediction-JMAN/notebooks/Model-training/model_comparison_summary.md)

### Best recorded result per model notebook

| Model | Variant | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---:|---:|---:|---:|---:|
| XGBoost | Tuned | 0.9565 | 0.8024 | 0.8074 | 0.8049 | 0.9806 |
| LightGBM | Tuned | 0.9549 | 0.7895 | 0.8110 | 0.8001 | 0.9808 |
| Gradient Boosting | Tuned | 0.9532 | 0.7754 | 0.8149 | 0.7947 | 0.9777 |
| Decision Tree | With SMOTE | 0.9432 | 0.7106 | 0.8256 | 0.7638 | N/A |
| AdaBoost | Tuned | 0.9353 | 0.6640 | 0.8477 | 0.7447 | 0.9711 |
| Random Forest | Without SMOTE | 0.9421 | 0.9037 | 0.5371 | 0.6738 | N/A |
| Logistic Regression | Baseline | 0.8909 | 0.5056 | 0.8571 | 0.6360 | 0.9529 |
| KNN | Tuned | 0.9187 | 0.6740 | 0.5221 | 0.5884 | 0.8845 |
| Naive Bayes | Baseline | 0.8743 | 0.4595 | 0.7356 | 0.5656 | 0.9028 |
| SVM | Tuned | 0.8508 | 0.4132 | 0.8118 | 0.5477 | 0.9157 |

### Overall ranking by F1

| Rank | Model | F1 |
|---:|---|---:|
| 1 | XGBoost | 0.8049 |
| 2 | LightGBM | 0.8001 |
| 3 | Gradient Boosting | 0.7947 |
| 4 | Decision Tree | 0.7638 |
| 5 | AdaBoost | 0.7447 |
| 6 | Random Forest | 0.6738 |
| 7 | Logistic Regression | 0.6360 |
| 8 | KNN | 0.5884 |
| 9 | Naive Bayes | 0.5656 |
| 10 | SVM | 0.5477 |

### Practical interpretation

- Best overall model: XGBoost
- Best ROC AUC: LightGBM
- Best recall: Logistic Regression
- Best precision: Random Forest without SMOTE
- Best balanced tree/boosting shortlist:
  - XGBoost
  - LightGBM
  - Gradient Boosting

## Final Recommendation

Based on the recorded notebook results:

- Use XGBoost as the primary production candidate.
- Keep LightGBM as the closest backup because it is nearly tied on F1 and slightly best on ROC AUC.
- Use Logistic Regression or AdaBoost as recall-oriented benchmark models when the business wants to catch as many churn cases as possible.

## How to Reproduce

Run the notebooks in this order:

1. Cleaning notebooks
2. Merge notebook
3. EDA notebooks
4. Hypothesis notebooks
5. Feature engineering notebook
6. Model training notebooks

Recommended sequence:

```text
notebooks/cleaning/01_billings.ipynb
notebooks/cleaning/02_cc_calls.ipynb
notebooks/cleaning/03_emails.ipynb
notebooks/cleaning/04_renewal_calls.ipynb
notebooks/Merge/01_merging.ipynb
notebooks/EDA/01_master_eda.ipynb
notebooks/hypothesis/04_hypothesis_merged.ipynb
notebooks/features/01_feature_engg.ipynb
notebooks/Model-training/*.ipynb
```

## Notes

- The project is notebook-first; `src/` is currently minimal.
- `requirements.txt` is still empty and should be filled if the project is being packaged for handoff.
- Some variables and feature choices are intentionally business-driven rather than purely automated, especially around leakage handling and feature aggregation.
