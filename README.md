# cardiovascular-disease-EDA
Exploratory Data Analysis (EDA), data preprocessing, statistical analysis, and feature engineering on a cardiovascular disease dataset using Python.

# ❤️ Cardiovascular Disease Exploratory Data Analysis (EDA)

A complete Exploratory Data Analysis (EDA) project on the **Cardiovascular Disease Dataset** using Python.

This project focuses on data cleaning, preprocessing, statistical analysis, visualization, and feature engineering to better understand the factors associated with cardiovascular disease.

---

## 📌 Project Objectives

The main objectives of this project are:

- Understand the dataset structure
- Detect duplicated records
- Check missing values
- Detect abnormal observations (Outliers)
- Apply medical knowledge for data cleaning
- Perform Exploratory Data Analysis (EDA)
- Perform statistical hypothesis testing
- Engineer new features (BMI)
- Prepare a clean dataset for future Machine Learning models

---

## 📊 Dataset

The dataset contains medical information collected from approximately **70,000 patients**.

### Dataset Features

| Feature | Description |
|----------|-------------|
| Age | Patient age (converted from days to years) |
| Gender | Male / Female |
| Height | Height (cm) |
| Weight | Weight (kg) |
| ap_hi | Systolic Blood Pressure |
| ap_lo | Diastolic Blood Pressure |
| Cholesterol | Cholesterol Level |
| Glucose | Glucose Level |
| Smoke | Smoking Status |
| Alcohol | Alcohol Consumption |
| Active | Physical Activity |
| Cardio | Cardiovascular Disease (Target) |

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

- Remove unnecessary columns
- Check missing values
- Detect duplicated records
- Remove duplicated samples
- Convert Age from days to years
- Optimize data types
- Export cleaned dataset

---

## 📈 Exploratory Data Analysis

EDA includes:

- Histogram
- Boxplot
- Countplot
- Scatter Plot
- Correlation Heatmap
- KDE Plot
- Boxen Plot

---

## ⚠️ Outlier Detection

Outliers were detected using the **Interquartile Range (IQR)** method.

However, outliers in:

- Age
- Height
- Weight

were retained because they may represent genuine physiological variations rather than measurement errors.

---

## 🩺 Medical Data Cleaning

Instead of relying only on statistical methods, medically impossible blood pressure values were removed.

Examples include:

- Systolic BP < 70
- Systolic BP > 250
- Diastolic BP < 40
- Diastolic BP > 150
- Systolic BP ≤ Diastolic BP

---

## 📊 Statistical Analysis

The following statistical tests were performed:

### Shapiro-Wilk Test

- Normality Assessment

### Chi-Square Test

- Relationship between categorical variables and cardiovascular disease

### Mann–Whitney U Test

- Comparison of numerical variables between healthy and diseased patients

---

## ⚙️ Feature Engineering

A new feature was created:

### Body Mass Index (BMI)

BMI = Weight / Height²

Patients were categorized into:

- Underweight
- Normal
- Overweight
- Obese

---

## 🔍 Key Findings

The analysis revealed that:

- Older individuals have a higher prevalence of cardiovascular disease.
- Higher systolic and diastolic blood pressure are strongly associated with cardiovascular disease.
- Obesity significantly increases disease prevalence.
- Elevated cholesterol is one of the strongest indicators.
- Elevated glucose levels are associated with cardiovascular disease.
- Physical inactivity is associated with higher disease occurrence.
- Smoking showed a weaker association in this dataset.

---

## 📁 Repository Structure

```
.
├── cardio_analysis.ipynb
├── cardio_analysis.py
├── cardio_train.csv
├── clean_cardio.csv
├── Cardiovascular_Disease_EDA_Report.pdf
├── README.md
├── LICENSE
```

---

## 🚀 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy

---

## 📄 Project Report

A detailed report describing the preprocessing, exploratory data analysis, statistical tests, and feature engineering is available in:

**Cardiovascular_Disease_EDA_Report.pdf**

---

## 👨‍💻 Author

**Reza Akbari-Hasanjani**

PhD in Electronic Engineering

Data Analysis | Python

---

## 📜 License

This project is licensed under the **MIT License**.

