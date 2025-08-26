# %% ================================
# 0) Imports & Settings
# ================================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
sns.set(style="whitegrid", palette="Set2")

# %% ================================
# 1) Load Data
# ================================
DATA_PATH = r"C:\Users\StefanSchäfer\OneDrive - rheindata GmbH\Learning\Projects\Python\Analystbuilder\Data\u_food_marketing.csv"
food = pd.read_csv(DATA_PATH)

print("Initial shape:", food.shape)
print("Duplicate rows:", food.duplicated().sum())
food.info()

# Drop duplicates
food.drop_duplicates(keep=False, inplace=True)
print("\nAfter dropping duplicates:", food.shape)

# %% ================================
# 2) Feature Engineering
# ================================

# --- Total children
food["total_children"] = food[["Kidhome", "Teenhome"]].sum(axis=1)

# --- Marital status
marital_map = {
    "marital_Widow": (1, "Widow"),
    "marital_Together": (2, "Together"),
    "marital_Single": (3, "Single"),
    "marital_Married": (4, "Married"),
    "marital_Divorced": (5, "Divorced"),
}

for col, (val, _) in marital_map.items():
    food[col] = food[col].replace({1: val, 0: 0})

food["marital_status"] = food[list(marital_map.keys())].sum(axis=1)
food["marital_status_str"] = food["marital_status"].replace({v: s for _, (v, s) in marital_map.items()})
food.drop(columns=marital_map.keys(), inplace=True)

# --- Education
edu_map = {
    "education_2n Cycle": (1, "2n Cycle"),
    "education_Basic": (2, "Basic"),
    "education_Graduation": (3, "Graduation"),
    "education_Master": (4, "Master"),
    "education_PhD": (5, "PhD"),
}

for col, (val, _) in edu_map.items():
    food[col] = food[col].replace({1: val, 0: 0})

food["education_status"] = food[list(edu_map.keys())].sum(axis=1)
food["education_status_str"] = food["education_status"].replace({v: s for _, (v, s) in edu_map.items()})
food.drop(columns=edu_map.keys(), inplace=True)

# --- Campaigns
food["accepted_campaigns"] = (
    food[["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5"]]
    .sum(axis=1)
    .gt(0)
    .astype(int)
)

# --- Age groups
food["age_group"] = pd.cut(
    food["Age"],
    bins=[0, 18, 30, 40, 50, 60, 70, 80, 90, 100],
    labels=["0-18", "19-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81-90", "91-100"],
)

# %% ================================
# 3) Correlation Analysis
# ================================
corrs = (
    food.select_dtypes(include="number")
    .corr(method="pearson")["accepted_campaigns"]
    .sort_values(ascending=False)
)
filtered_corrs = corrs[(corrs > 0.3) & (corrs < 1)]
print(filtered_corrs)

# %% ================================
# 4) Helper Functions for Plotting
# ================================
def plot_bar(df, x, y, title, palette="Set2"):
    sns.barplot(data=df, x=x, y=y, palette=palette)
    plt.title(title)
    plt.show()

def plot_reg(df, x, y, title=""):
    sns.regplot(data=df, x=x, y=y)
    plt.title(title)
    plt.show()

# %% ================================
# 5) Analysis
# ================================

# Spending by age group
spending_age = food.groupby("age_group")["MntTotal"].sum().reset_index()
spending_age = spending_age[spending_age["MntTotal"] > 0].copy()
spending_age["age_group"] = spending_age["age_group"].cat.remove_unused_categories()
plot_bar(spending_age, "age_group", "MntTotal", "Total Spend by Age Group")

# Spending by marital status
spending_marital = food.groupby("marital_status_str")["MntTotal"].sum().reset_index()
plot_bar(spending_marital, "marital_status_str", "MntTotal", "Spend by Marital Status")

# Campaign acceptance vs. children
plot_reg(food, "total_children", "accepted_campaigns", "Campaigns Accepted vs. Children")

# Campaign acceptance vs. education
plot_reg(food, "education_status", "accepted_campaigns", "Campaigns Accepted vs. Education")

# %% ================================
# 6) Insights (to refine)
# ================================
"""
Findings:
1. Age 31–70 → highest spenders, but less campaign acceptance.
2. Catalog users → accept campaigns more; in-store/web → higher total spend.
   Suggested split: 40% catalog, 30% store, 30% web.
3. Fewer children → higher spend and higher acceptance.
4. Education → little impact.
5. Marital status → Married, Single, Together groups drive most revenue.
"""