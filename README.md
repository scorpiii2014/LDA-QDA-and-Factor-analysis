# 📊 LDA, QDA and Factor Analysis on Sales Product Details

## 📌 Project Overview

This project applies statistical and machine learning techniques — 
**Linear Discriminant Analysis (LDA)**, 
**Quadratic Discriminant Analysis (QDA)**, and 
**Factor Analysis** — on retail sales data to perform profit classification 
and uncover underlying business factors influencing performance.

The analysis is implemented using **R programming language**.

---

## 🎯 Objectives

- Classify transactions into High Profit and Low Profit categories
- Compare LDA and QDA model performance
- Identify latent factors influencing sales performance
- Provide business insights using statistical modeling

---

## 📂 Dataset

Dataset used: Superstore Sales Dataset  
Features include:

- Sales
- Profit
- Quantity
- Discount
- Category
- Sub-Category
- Region
- Segment

---

## 🛠️ Technologies & Libraries Used

- R
- MASS (for LDA & QDA)
- psych (for Factor Analysis)
- corrplot
- caret
- ggplot2

---

## 📈 Methodology

### 1️⃣ Data Preprocessing
- Data cleaning
- Feature selection
- Conversion of Profit into binary classification variable

### 2️⃣ Linear Discriminant Analysis (LDA)
- Assumes equal covariance matrices
- Creates linear decision boundary
- Evaluated using confusion matrix and accuracy

### 3️⃣ Quadratic Discriminant Analysis (QDA)
- Allows different covariance matrices
- Creates quadratic decision boundary
- Performance comparison with LDA

### 4️⃣ Factor Analysis
- KMO Test
- Bartlett’s Test
- Extraction of latent factors
- Interpretation of factor loadings

---

## 📊 Results

- QDA showed improved classification accuracy over LDA.
- Discount significantly impacts profitability.
- Two major latent factors were identified:
  - Revenue Performance Factor
  - Pricing & Volume Strategy Factor

---

## 💡 Key Business Insights

- High discounts negatively affect profit margins.
- Sales and quantity are strong predictors of profitability.
- Dimensionality reduction helps simplify decision-making variables.

---

## 📁 Project Structure

