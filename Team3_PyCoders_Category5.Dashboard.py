import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

sns.set_style("whitegrid")




st.set_page_config(
    page_title="Heart Failure Care Dashboard",
    page_icon="❤️",
    layout="wide"
)
PRIMARY = "#3F8F87"
SECONDARY = "#315F6B"
LIGHT_TEAL = "#76B7B2"
ACCENT = "#E6A15A"
DARK = "#1F2937"

st.markdown("""
<style>

/* Main dashboard title */
h1 {
    color: #1F2937 !important;
    -webkit-text-fill-color: #1F2937 !important;
}

/* Section titles */
h2, h3 {
    color: #315F6B !important;
}

/* KPI cards */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #FFFFFF, #F0FDFA);
    border: 1.5px solid #76B7B2;
    border-left: 8px solid #3F8F87;
    border-radius: 20px;
    padding: 22px 24px;
    box-shadow: 0 5px 15px rgba(63,143,135,0.15);
}

/* KPI label */
div[data-testid="stMetricLabel"] {
    font-size: 18px;
    font-weight: 500;
    color: #374151;
}

/* KPI number */
div[data-testid="stMetricValue"] {
    font-size: 46px;
    font-weight: 800;
    color: #3F8F87;
}

/* Male/Female patient count */
div[data-testid="stMetricDelta"] {
    background: #E8F5F2;
    color: #28766F !important;
    padding: 5px 10px;
    border-radius: 18px;
    width: fit-content;
    font-size: 16px;
}

/* Dataframes */
div[data-testid="stDataFrame"] {
    border: 1px solid #76B7B2;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)



# LOAD DATA


@st.cache_data
def load_data():

    df = pd.read_csv("cleaned_files/heart_failure_merged.csv")

    text_cols = [
        "patient_id",
        "gender",
        "age_category",
        "occupation",
        "discharge_destination",
        "admission_ward",
        "admission_mode",
        "discharge_department",
        "in_hosp_outcome"
    ]

    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace("nan", np.nan)

    numeric_cols = [
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
        "cci_score"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            ).fillna(0)

    outcome_cols = [
        "readmission_28d",
        "readmission_3mo",
        "readmission_6mo",
        "mortality_28d",
        "death_3mo"
    ]

    available = [
        c for c in outcome_cols
        if c in df.columns
    ]

    if available:
        df["poor_outcome"] = df[available].max(axis=1)
    else:
        df["poor_outcome"] = 0

    # Risk band
    df["risk_band"] = np.select(
        [
            df["poor_outcome"] == 0,

            (
                (df["poor_outcome"] == 1)
                &
                df[
                    "nyha_cardiac_function_classification"
                ].between(1, 3)
            ),

            (
                (df["poor_outcome"] == 1)
                &
                (
                    df[
                        "nyha_cardiac_function_classification"
                    ] >= 3
                )
            ),

            (
                (df["mod_severe_ckd"] == 1)
                |
                (df["diabetes"] == 1)
                |
                (df["myocardial_infarction"] == 1)
            )
        ],

        [
            "Low",
            "Moderate",
            "High",
            "Watchlist"
        ],

        default="Moderate"
    )

    return df



@st.cache_data
def get_summary(df):

    return pd.DataFrame({

        "Metric": [
            "Patients",
            "28-day readmission",
            "3-month readmission",
            "6-month readmission",
            "28-day mortality",
            "Poor outcome patients"
        ],

        "Value": [
            len(df),
            round(df["readmission_28d"].mean() * 100,1),

            round(df["readmission_3mo"].mean() * 100,1),

            round(df["readmission_6mo"].mean() * 100,),

            round(df["mortality_28d"].mean() * 100,1),

            int(df["poor_outcome"].sum())
        ]
    })


def format_pct(value):
    return f"{value:.1f}%"



def chart_time_trend(df):

    return pd.DataFrame({

        "Time": [
            "28 days",
            "3 months",
            "6 months"
        ],

        "Readmission rate (%)": [
            df["readmission_28d"].mean() * 100,
            df["readmission_3mo"].mean() * 100,
            df["readmission_6mo"].mean() * 100
        ]
    })




def chart_risk_factor(df):

    conditions = {
        "Diabetes": "diabetes",
        "CKD": "mod_severe_ckd",
        "Myocardial infarction": "myocardial_infarction",
        "Congestive HF": "congestive_heart_failure"
    }

    comparisons = []

    for label, col in conditions.items():

        if col in df.columns:

            rate = (
                df.groupby(col)["poor_outcome"]
                .mean()
                .reindex([0, 1], fill_value=0)
                * 100
            )

            comparisons.append({
                "Condition": label,
                "No": rate.get(0, 0),
                "Yes": rate.get(1, 0)
            })

    return pd.DataFrame(comparisons)

df = load_data()
st.title(
    "❤️ Heart Failure Risk & Discharge Planning Dashboard"
)

st.caption(
    "Local decision-support dashboard for prioritising "
    "follow-up, discharge review, and high-risk intervention."
)


#sidebar

st.sidebar.header("🔎 Filters")


selected_gender = st.sidebar.multiselect(
    "Gender",
    sorted(df["gender"].dropna().unique())
)


selected_age = st.sidebar.multiselect(
    "Age group",
    sorted(df["age_category"].dropna().unique())
)


selected_nyha = st.sidebar.multiselect(
    "NYHA class",
    sorted(
        df[
            "nyha_cardiac_function_classification"
        ]
        .dropna()
        .unique()
        .astype(int)
    )
)


selected_focus = st.sidebar.selectbox(
    "Clinical focus",
    [
        "All patients",
        "Diabetes",
        "CKD",
        "History of MI",
        "High NYHA / severe HF",
        "High CCI"
    ]
)

filtered = df.copy()


if selected_gender:
    filtered = filtered[
        filtered["gender"].isin(selected_gender)
    ]


if selected_age:
    filtered = filtered[
        filtered["age_category"].isin(selected_age)
    ]


if selected_nyha:
    filtered = filtered[
        filtered[
            "nyha_cardiac_function_classification"
        ].isin(selected_nyha)
    ]


if selected_focus == "Diabetes":

    filtered = filtered[
        filtered["diabetes"] == 1
    ]


elif selected_focus == "CKD":

    filtered = filtered[
        filtered["mod_severe_ckd"] == 1
    ]


elif selected_focus == "History of MI":

    filtered = filtered[
        filtered["myocardial_infarction"] == 1
    ]


elif selected_focus == "High NYHA / severe HF":

    filtered = filtered[
        filtered[
            "nyha_cardiac_function_classification"
        ] >= 3
    ]


elif selected_focus == "High CCI":

    filtered = filtered[
        filtered["cci_score"] >= 3
    ]


st.sidebar.caption(
    f"Current cohort: {len(filtered):,} patients"
)


if filtered.empty:

    st.warning(
        "No patients match the selected filters."
    )

    st.stop()




summary = get_summary(filtered)

st.subheader("📌 Patient Overview")

c1, c2, c3, c4 = st.columns(4)


c1.metric(
    "👥 Patients",
    f"{int(summary.loc[0, 'Value']):,}"
)


c2.metric(
    "🔄 28-Day Readmission",
    format_pct(summary.loc[1, "Value"])
)


c3.metric(
    "🗓️ 3-Month Readmission",
    format_pct(summary.loc[2, "Value"])
)


c4.metric(
    "🏥 6-Month Readmission",
    format_pct(summary.loc[3, "Value"])
)




gender_clean = (
    filtered["gender"]
    .astype(str)
    .str.strip()
    .str.lower()
)


male_count = gender_clean.isin(
    ["male", "m"]
).sum()


female_count = gender_clean.isin(
    ["female", "f"]
).sum()


total_gender = male_count + female_count


male_pct = (
    male_count / total_gender * 100
    if total_gender else 0
)


female_pct = (
    female_count / total_gender * 100
    if total_gender else 0
)


male_col, female_col = st.columns(2)


male_col.metric(
    "👨 Male",
    f"{male_pct:.1f}%",
    f"{male_count:,} patients"
)


female_col.metric(
    "👩 Female",
    f"{female_pct:.1f}%",
    f"{female_count:,} patients"
)


st.markdown("---")




left, right = st.columns(2)
with left:

    st.subheader("🫀 Severity Profile")

    nyha = (
        filtered[
            "nyha_cardiac_function_classification"
        ]
        .value_counts()
        .sort_index()
    )

    colors = [
        "#76B7B2",
        "#4E8F9E",
        "#315F6B",
        "#E6A15A"
    ][:len(nyha)]

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    for x, y, color in zip(
        nyha.index,
        nyha.values,
        colors
    ):

        ax.vlines(
            x,
            0,
            y,
            color=color,
            linewidth=5
        )

        ax.scatter(
            x,
            y,
            color=color,
            s=220
        )

        ax.text(
            x,
            y + 45,
            str(y),
            ha="center",
            fontweight="bold"
        )

    ax.set_title(
        "Patients by NYHA Class"
    )

    ax.set_xlabel("NYHA Class")
    ax.set_ylabel("Patients")

    ax.set_xticks(nyha.index)

    ax.set_ylim(
        0,
        nyha.max() * 1.18
    )

    sns.despine()

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)
with right:

    st.subheader(
        "👥 Population Breakdown"
    )

    age_group = (
        filtered["age_category"]
        .value_counts()
        .head(10)
    )

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    sns.barplot(
        x=age_group.values,
        y=age_group.index,
        hue=age_group.index,
        palette="crest",
        legend=False,
        ax=ax
    )

    ax.set_title(
        "Patients by Age Category"
    )

    ax.set_xlabel("Patients")
    ax.set_ylabel("Age Group")

    for container in ax.containers:

        ax.bar_label(
            container,
            padding=3
        )

    sns.despine()

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)



# HIGH-RISK FACTORS


st.subheader(
    "⚠️ High-Risk Factor Comparison"
)

risk_df = chart_risk_factor(
    filtered
)


if not risk_df.empty:

    melted = risk_df.melt(
        id_vars="Condition",
        var_name="Group",
        value_name="Poor outcome rate (%)"
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    sns.barplot(
        data=melted,
        x="Condition",
        y="Poor outcome rate (%)",
        hue="Group",
        palette=[
            "#76B7B2",
            "#E6A15A"
        ],
        ax=ax
    )

    ax.set_title(
        "Poor Outcome Rate by Clinical Risk Factor"
    )

    ax.set_xlabel("")

    ax.set_ylabel(
        "Poor Outcome Rate (%)"
    )

    for container in ax.containers:

        ax.bar_label(
            container,
            fmt="%.1f%%",
            padding=3
        )

    ax.legend(
        title="Condition Present"
    )

    sns.despine()

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


st.markdown("---")
left, right = st.columns(2)
with left:

    st.subheader(
        "🏥 Ward & Discharge"
    )

    pivot = filtered.pivot_table(
        values="readmission_28d",
        index="admission_ward",
        columns="discharge_destination",
        aggfunc="mean"
    ) * 100

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        linewidths=.5,
        ax=ax
    )

    ax.set_title(
        "Readmission by Admission Ward\n"
        "and Discharge Destination"
    )

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)



# CORRELATION
with right:

    st.subheader(
        "📊 Clinical Correlations"
    )

    corr_cols = [
        "cci_score",
        "gfr",
        "discharge_day",
        "pulse",
        "systolic_bp",
        "respiration",
        "gcs",
        "nyha_cardiac_function_classification",
        "killip_grade",
        "lvef",
        "readmission_28d",
        "readmission_6mo"
    ]

    corr_cols = [
        c for c in corr_cols
        if c in filtered.columns
    ]

    corr = (
        filtered[corr_cols]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .corr()
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        linewidths=.3,
        annot_kws={"size": 6},
        ax=ax
    )

    ax.set_title(
        "Clinical Markers & Readmission Correlations"
    )

    ax.tick_params(
        axis="x",
        labelrotation=90,
        labelsize=6
    )

    ax.tick_params(
        axis="y",
        labelsize=6
    )

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


st.markdown("---")
left, right = st.columns(
    [1.5, 1]
)
with left:

    st.subheader(
        "📈 Readmission Over Time"
    )

    trend = chart_time_trend(
        filtered
    )

    fig, ax = plt.subplots(
        figsize=(6, 3.5)
    )

    sns.lineplot(
        data=trend,
        x="Time",
        y="Readmission rate (%)",
        marker="o",
        markersize=10,
        linewidth=3,
        color=PRIMARY,
        ax=ax
    )

    for i, row in (
        trend
        .reset_index(drop=True)
        .iterrows()
    ):

        ax.annotate(
            f'{row["Readmission rate (%)"]:.1f}%',

            (
                i,
                row["Readmission rate (%)"]
            ),

            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontweight="bold"
        )

    ax.set_title(
        "Readmission Trend"
    )

    ax.set_xlabel(
        "Follow-up Period"
    )

    ax.set_ylabel(
        "Readmission Rate (%)"
    )

    maximum = (
        trend[
            "Readmission rate (%)"
        ].max()
    )

    ax.set_ylim(
        0,
        maximum * 1.2
        if maximum > 0
        else 10
    )

    sns.despine()

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)



# KEY CLINICAL OUTCOMES


with right:

    st.subheader(
        "📋 Key Clinical Outcomes"
    )

    outcome_table = pd.DataFrame({

        "Outcome": [
            "28-day mortality",
            "Poor outcome patients",
            "High-risk watchlist"
        ],

        "Rate / count": [

            format_pct(
                summary.loc[4, "Value"]
            ),

            f"{summary.loc[5, 'Value']:,}",

            f"{int(
                (
                    filtered["risk_band"]
                    == "Watchlist"
                ).sum()
            ):,}"
        ]
    })

    st.dataframe(
        outcome_table,
        hide_index=True,
        use_container_width=True
    )


    # CKD rate
    ckd = filtered[
        filtered["mod_severe_ckd"] == 1
    ]

    ckd_rate = (
        ckd["readmission_28d"].mean() * 100
        if len(ckd)
        else 0
    )

    st.write(
        f"🩺 CKD 28-day readmission: "
        f"**{ckd_rate:.1f}%**"
    )


    # Diabetes rate
    diabetes = filtered[
        filtered["diabetes"] == 1
    ]

    diabetes_rate = (
        diabetes["readmission_28d"].mean() * 100
        if len(diabetes)
        else 0
    )

    st.write(
        f"💉 Diabetes 28-day readmission: "
        f"**{diabetes_rate:.1f}%**"
    )


st.markdown("---")



# HIGH-PRIORITY PATIENT LIST
st.subheader(
    "🚨 High-Priority Patient List"
)

priority = filtered.copy()


priority["total_risk_score"] = (

    priority.get(
        "readmission_28d", 0
    )

    +

    priority.get(
        "readmission_3mo", 0
    )

    +

    priority.get(
        "readmission_6mo", 0
    )

    +

    priority.get(
        "mortality_28d", 0
    )

    +

    priority.get(
        "death_3mo", 0
    )

    +

    priority.get(
        "diabetes", 0
    )

    +

    priority.get(
        "mod_severe_ckd", 0
    )

    +

    priority.get(
        "myocardial_infarction", 0
    )

    +

    (
        priority[
            "nyha_cardiac_function_classification"
        ] >= 3
    ).astype(int)
)


priority = (
    priority
    .sort_values(
        "total_risk_score",
        ascending=False
    )
    .head(20)
)


priority_cols = [
    "patient_id",
    "gender",
    "age_category",
    "nyha_cardiac_function_classification",
    "killip_grade",
    "diabetes",
    "mod_severe_ckd",
    "myocardial_infarction",
    "cci_score",
    "poor_outcome"
]


priority_cols = [
    c for c in priority_cols
    if c in priority.columns
]


priority_table = (
    priority[
        priority_cols
    ]
    .copy()
)


priority_table.rename(

    columns={

        "patient_id":
        "Patient ID",

        "gender":
        "Gender",

        "age_category":
        "Age group",

        "nyha_cardiac_function_classification":
        "NYHA",

        "killip_grade":
        "Killip",

        "diabetes":
        "Diabetes",

        "mod_severe_ckd":
        "CKD",

        "myocardial_infarction":
        "MI",

        "cci_score":
        "CCI",

        "poor_outcome":
        "Poor outcome"
    },

    inplace=True
)


st.dataframe(
    priority_table,
    use_container_width=True,
    hide_index=True
)


st.markdown("---")



# RECOMMENDED ACTION PLAN


st.subheader(
    "💡 Recommended Action Plan"
)
risk_flags = []
if (
    filtered[
        "mod_severe_ckd"
    ].mean() > .2
):

    risk_flags.append(
        "Patients with moderate-to-severe CKD "
        "are a priority for discharge medication "
        "review and follow-up scheduling."
    )


if (
    filtered[
        "diabetes"
    ].mean() > .2
):

    risk_flags.append(
        "Diabetes is common in the current cohort; "
        "prioritize glucose management education "
        "and post-discharge check-ins."
    )


if (
    filtered[
        "myocardial_infarction"
    ].mean() > .05
):

    risk_flags.append(
        "History of myocardial infarction should "
        "trigger focused review for early "
        "readmission risk."
    )


severe_share = (

    (
        filtered[
            "nyha_cardiac_function_classification"
        ] >= 3
    ).mean()

    * 100
)


if severe_share > 40:

    risk_flags.append(

        f"{severe_share:.1f}% of the selected "
        "cohort is NYHA class 3–4; consider "
        "closer monitoring and earlier follow-up."
    )


if risk_flags:

    for item in risk_flags:
        st.info(item)

else:

    st.success(
        "The selected cohort does not show a "
        "dominant risk profile; continue routine "
        "follow-up with targeted review for "
        "flagged patients."
    )