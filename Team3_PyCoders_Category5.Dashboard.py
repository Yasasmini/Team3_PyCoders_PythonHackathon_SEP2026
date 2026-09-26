import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

sns.set_style("whitegrid")


@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_files/heart_failure_merged.csv")

    for col in [
        "patient_id",
        "gender",
        "age_category",
        "occupation",
        "discharge_destination",
        "admission_ward",
        "admission_mode",
        "discharge_department",
        "in_hosp_outcome",
    ]:
        if col in df.columns:
            df[col] = df[col].astype(str).replace("nan", np.nan)

    for col in [
        "readmission_28d",
        "readmission_3mo",
        "readmission_6mo",
        "mortality_28d",
        "death_3mo",
        "diabetes",
        "mod_severe_ckd",
        "myocardial_infarction",
        "congestive_heart_failure",
        "copd",
        "dementia",
        "nyha_cardiac_function_classification",
        "killip_grade",
        "cci_score",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    outcome_cols = [
        "readmission_28d",
        "readmission_3mo",
        "readmission_6mo",
        "mortality_28d",
        "death_3mo",
    ]
    if all(col in df.columns for col in outcome_cols):
        df["poor_outcome"] = df[outcome_cols].max(axis=1)
    else:
        df["poor_outcome"] = 0

    if "readmission_28d" in df.columns:
        df["risk_band"] = np.select(
            [
                df["poor_outcome"] == 0,
                (df["poor_outcome"] == 1) & (df["nyha_cardiac_function_classification"].between(1, 3)),
                (df["poor_outcome"] == 1) & (df["nyha_cardiac_function_classification"].ge(3)),
                (df["mod_severe_ckd"] == 1) | (df["diabetes"] == 1) | (df["myocardial_infarction"] == 1),
            ],
            ["Low", "Moderate", "High", "Watchlist"],
            default="Moderate",
        )
    else:
        df["risk_band"] = "Moderate"

    return df


@st.cache_data
def get_summary(df):
    summary = {
        "Metric": [
            "Patients",
            "28-day readmission",
            "3-month readmission",
            "6-month readmission",
            "28-day mortality",
            "Poor outcome patients",
        ],
        "Value": [
            len(df),
            round(df["readmission_28d"].mean() * 100, 1) if "readmission_28d" in df.columns else 0,
            round(df["readmission_3mo"].mean() * 100, 1) if "readmission_3mo" in df.columns else 0,
            round(df["readmission_6mo"].mean() * 100, 1) if "readmission_6mo" in df.columns else 0,
            round(df["mortality_28d"].mean() * 100, 1) if "mortality_28d" in df.columns else 0,
            int(df["poor_outcome"].sum()) if "poor_outcome" in df.columns else 0,
        ],
    }
    return pd.DataFrame(summary)


st.set_page_config(page_title="Heart Failure Care Dashboard", page_icon="❤️", layout="wide")


def format_pct(value):
    return f"{value:.1f}%"


def chart_time_trend(df):
    outcome_labels = ["28 days", "3 months", "6 months"]
    outcome_cols = ["readmission_28d", "readmission_3mo", "readmission_6mo"]
    values = []
    for col in outcome_cols:
        if col in df.columns:
            values.append(df[col].mean() * 100)
        else:
            values.append(0)

    trend = pd.DataFrame({"Time": outcome_labels, "Readmission rate (%)": values})
    return trend


def chart_risk_factor(df):
    comparisons = []
    conditions = {
        "Diabetes": "diabetes",
        "CKD": "mod_severe_ckd",
        "Myocardial infarction": "myocardial_infarction",
        "Congestive HF": "congestive_heart_failure",
    }

    for label, col in conditions.items():
        if col in df.columns:
            rate = df.groupby(col)["poor_outcome"].mean().reindex([0, 1], fill_value=0) * 100
            comparisons.append({
                "Condition": label,
                "No": rate.get(0, 0),
                "Yes": rate.get(1, 0),
            })

    if comparisons:
        return pd.DataFrame(comparisons)
    return pd.DataFrame({"Condition": [], "No": [], "Yes": []})


# App layout
df = load_data()

st.title("❤️ Heart Failure Risk & Discharge Planning Dashboard")
st.caption("Local decision-support dashboard for prioritising follow-up, discharge review, and high-risk intervention.")

st.sidebar.header("Filters")
selected_gender = st.sidebar.multiselect("Gender", sorted(df["gender"].dropna().unique()))
selected_age = st.sidebar.multiselect("Age group", sorted(df["age_category"].dropna().unique()))
selected_nyha = st.sidebar.multiselect(
    "NYHA class",
    sorted(df["nyha_cardiac_function_classification"].dropna().unique().astype(int)),
)
selected_focus = st.sidebar.selectbox(
    "Clinical focus",
    ["All patients", "Diabetes", "CKD", "History of MI", "High NYHA / severe HF", "High CCI"],
)

filtered = df.copy()
if selected_gender:
    filtered = filtered[filtered["gender"].isin(selected_gender)]
if selected_age:
    filtered = filtered[filtered["age_category"].isin(selected_age)]
if selected_nyha:
    filtered = filtered[filtered["nyha_cardiac_function_classification"].isin(selected_nyha)]

if selected_focus == "Diabetes":
    filtered = filtered[filtered["diabetes"] == 1]
elif selected_focus == "CKD":
    filtered = filtered[filtered["mod_severe_ckd"] == 1]
elif selected_focus == "History of MI":
    filtered = filtered[filtered["myocardial_infarction"] == 1]
elif selected_focus == "High NYHA / severe HF":
    filtered = filtered[filtered["nyha_cardiac_function_classification"] >= 3]
elif selected_focus == "High CCI":
    filtered = filtered[filtered["cci_score"] >= 3]

st.sidebar.caption(f"Current cohort: {len(filtered)} patients")

summary = get_summary(filtered)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Patients", f"{int(summary.loc[0, 'Value']):,}")
col2.metric("28-day readmission", format_pct(summary.loc[1, 'Value']))
col3.metric("3-month readmission", format_pct(summary.loc[2, 'Value']))
col4.metric("6-month readmission", format_pct(summary.loc[3, 'Value']))

st.markdown("---")

left, right = st.columns([1.5, 1])
with left:
    st.subheader("Readmission over time")
    trend = chart_time_trend(filtered)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=trend, x="Time", y="Readmission rate (%)", palette="Blues_d", ax=ax)
    ax.set_ylabel("Readmission rate (%)")
    ax.set_xlabel("Follow-up window")
    ax.set_title("Readmission trend")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom")
    st.pyplot(fig)

with right:
    st.subheader("Key clinical outcomes")
    outcome_table = pd.DataFrame(
        {
            "Outcome": ["28-day mortality", "Poor outcome patients", "High-risk watchlist"],
            "Rate / count": [
                format_pct(summary.loc[4, "Value"]),
                f"{summary.loc[5, 'Value']:,}",
                f"{int((filtered['risk_band'] == 'Watchlist').sum()):,}",
            ],
        }
    )
    st.dataframe(outcome_table, hide_index=True, use_container_width=True)

    if "mortality_28d" in filtered.columns:
        ckd_rate = (filtered[filtered["mod_severe_ckd"] == 1]["readmission_28d"].mean() * 100) if "mod_severe_ckd" in filtered.columns else 0
        diabetes_rate = (filtered[filtered["diabetes"] == 1]["readmission_28d"].mean() * 100) if "diabetes" in filtered.columns else 0
        st.write(f"CKD 28-day readmission rate: {ckd_rate:.1f}%")
        st.write(f"Diabetes 28-day readmission rate: {diabetes_rate:.1f}%")

st.markdown("---")

st.subheader("High-risk factor comparison")

risk_df = chart_risk_factor(filtered)
if not risk_df.empty:
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    melted = risk_df.melt(id_vars="Condition", var_name="Group", value_name="Poor outcome rate (%)")
    sns.barplot(data=melted, x="Condition", y="Poor outcome rate (%)", hue="Group", palette=["#7ec8e3", "#ff7f7f"], ax=ax2)
    ax2.set_title("Poor outcome rate by clinical risk flag")
    ax2.set_xlabel("Condition")
    ax2.set_ylabel("Poor outcome rate (%)")
    ax2.legend(title="Group")
    st.pyplot(fig2)
else:
    st.info("No comparable clinical risk data available for the active filter set.")

st.markdown("---")

left2, right2 = st.columns(2)
with left2:
    st.subheader("Severity profile")
    if "nyha_cardiac_function_classification" in filtered.columns:
        nyha_summary = (
            filtered["nyha_cardiac_function_classification"]
            .value_counts()
            .sort_index()
            .rename_axis("NYHA class")
            .reset_index(name="Patients")
        )
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        sns.barplot(data=nyha_summary, x="NYHA class", y="Patients", palette="viridis", ax=ax3)
        ax3.set_title("Patient count by NYHA class")
        st.pyplot(fig3)

with right2:
    st.subheader("Population breakdown")
    if not filtered.empty:
        age_group = filtered["age_category"].value_counts().head(10)
        fig4, ax4 = plt.subplots(figsize=(7, 4))
        sns.barplot(x=age_group.values, y=age_group.index, orient="h", palette="magma", ax=ax4)
        ax4.set_title("Patients by age category")
        ax4.set_xlabel("Patients")
        ax4.set_ylabel("Age group")
        st.pyplot(fig4)

st.markdown("---")

st.subheader("High-priority patient list")
priority = filtered.copy()
priority["total_risk_score"] = (
    priority.get("readmission_28d", 0)
    + priority.get("readmission_3mo", 0)
    + priority.get("readmission_6mo", 0)
    + priority.get("mortality_28d", 0)
    + priority.get("death_3mo", 0)
    + priority.get("diabetes", 0)
    + priority.get("mod_severe_ckd", 0)
    + priority.get("myocardial_infarction", 0)
    + (priority["nyha_cardiac_function_classification"] >= 3).astype(int)
)
priority = priority.sort_values("total_risk_score", ascending=False).head(20)

priority_table = priority[
    [
        "patient_id",
        "gender",
        "age_category",
        "nyha_cardiac_function_classification",
        "killip_grade",
        "diabetes",
        "mod_severe_ckd",
        "myocardial_infarction",
        "cci_score",
        "poor_outcome",
    ]
].copy()
priority_table.columns = [
    "Patient ID",
    "Gender",
    "Age group",
    "NYHA",
    "Killip",
    "Diabetes",
    "CKD",
    "MI",
    "CCI",
    "Poor outcome",
]

st.dataframe(priority_table, use_container_width=True, hide_index=True)

st.markdown("---")

st.subheader("Recommended action plan")

risk_flags = []
if "mod_severe_ckd" in filtered.columns and filtered["mod_severe_ckd"].mean() > 0.2:
    risk_flags.append("Patients with moderate-to-severe CKD are a priority for discharge medication review and follow-up scheduling.")
if "diabetes" in filtered.columns and filtered["diabetes"].mean() > 0.2:
    risk_flags.append("Diabetes is common in the current cohort; prioritize glucose management education and post-discharge check-ins.")
if "myocardial_infarction" in filtered.columns and filtered["myocardial_infarction"].mean() > 0.05:
    risk_flags.append("History of myocardial infarction should trigger a focused case review for early readmission risk.")
if "nyha_cardiac_function_classification" in filtered.columns:
    severe_share = (filtered["nyha_cardiac_function_classification"] >= 3).mean() * 100
    if severe_share > 40:
        risk_flags.append(f"{severe_share:.1f}% of the selected cohort is in NYHA class 3-4; intensify monitoring and early follow-up.")

if risk_flags:
    for item in risk_flags:
        st.info(item)
else:
    st.success("The selected cohort does not show a dominant risk profile; continue routine follow-up with targeted review for any flagged patients.")


