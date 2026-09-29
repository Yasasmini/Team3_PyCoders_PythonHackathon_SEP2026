# Heart Failure Risk and Readmission Analysis

## 1. Project Description

This project analyzes clinical heart failure dataset data from 2,008 hospitalized heart-failure patients treated at Zigong Fourth People’s Hospital, China, between December 2016 and June 2019. The main focus is on identifying patients most likely to experience readmission or mortality after hospitalization.

The dataset includes multiple components such as:
- patient demographics
- comorbidity history
- cardiac severity indicators
- hospital admission and discharge information
- laboratory values
- medication profiles
- follow-up outcomes such as readmission and mortality

The project combines data cleaning, descriptive analysis, prescriptive analysis, predictive analysis, and a local Streamlit dashboard to support clinical decision-making and discharge planning.

## 2. Problem Statement

Heart failure is a high-burden condition associated with frequent readmissions, significant clinical complexity, and elevated mortality risk after discharge. In healthcare systems, limited clinical resources make it difficult to provide equally intensive follow-up to all patients.

The core problem addressed in this project is:

How can hospitals identify the patients most at risk of poor outcomes after discharge so that follow-up care, medication review, and monitoring can be prioritized effectively?

This project aims to support better care planning by identifying patterns in the data that can help clinicians focus on the most vulnerable patient groups.

## 3. Methods and Analysis

The project was implemented in multiple stages:

### Data Cleaning
- Merged multiple cleaned datasets into a unified heart failure dataset
- Standardized column names and data types
- Handled missing values and invalid entries
- Converted categorical/indicator fields to consistent formats
- Built a consolidated dataset suitable for downstream analysis

### Descriptive Analysis
- Explored patient demographics and age distribution
- Reviewed heart failure severity by NYHA and Killip grades
- Analyzed comorbidity burden and clinical patterns
- Assessed readmission and mortality trends across follow-up periods

### Prescriptive Analysis
- Compared readmission rates across clinically relevant subgroups
- Evaluated risk factors such as:
  - moderate-to-severe CKD
  - diabetes
  - myocardial infarction history
  - severe heart failure classification
- Identified patient groups that should receive targeted discharge review and follow-up coordination

### Predictive Analysis
- The predictive analysis evaluated which clinical features, such as NYHA severity, kidney function, and comorbidity burden, were associated with short-term readmission risk.
- The model indicated that these factors contain useful risk signals, but the predictive performance was limited, suggesting the need for additional variables or stronger modeling approaches.

### Dashboard
- Built a local Streamlit dashboard to summarize patient risk and outcomes
- Enabled interactive filtering by gender, age group, severity, and focus area
- Included key metrics and a high-priority patient view to support clinical review

## 4. Key Findings

The analysis highlighted several important clinical patterns:

- Readmission rates increased substantially over time:
  - 28-day readmission: approximately 7%
  - 3-month readmission: approximately 25%
  - 6-month readmission: approximately 38%

- Mortality also increased over time, emphasizing the importance of continued monitoring beyond the initial discharge period.

- Patients with moderate-to-severe CKD had noticeably higher early readmission rates than those without CKD.

- Patients with diabetes showed a higher risk of readmission compared with patients without diabetes.

- History of myocardial infarction was associated with a moderately elevated 28-day readmission risk.

- A large proportion of the cohort had severe heart failure classification (NYHA class 3 or 4), which suggests a need for more focused early follow-up and discharge planning.

- The strongest operational insight was that risk is not evenly distributed across the cohort; patients with comorbidity burden and severe functional limitation should be prioritized for intervention.

## 5. Team Members

- Yasasmini Vella
- Deepthi Kammarinaga
- Pavani Gujjula
- Ramyakrishna Yarram

## 6. Project Outcome

This project demonstrates a practical analytics workflow for improving heart failure patient care using data-driven prioritization. The cleaned dataset, analytical findings, and dashboard provide a strong foundation for informed clinical review, risk-based intervention, and quality-improvement decision-making.
