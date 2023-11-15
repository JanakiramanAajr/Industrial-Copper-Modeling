# Industrial-Copper-Modeling
## Project Overview

### Project Title
Industrial Copper Modeling

### Skills Takeaway
- Python scripting
- Data Preprocessing
- Exploratory Data Analysis (EDA)
- Streamlit
- Machine Learning (Regression and Classification)
- Model Deployment
- GitHub Collaboration
- Video Demonstration

### Domain
Manufacturing

## Problem Statement

The copper industry faces challenges in sales and pricing due to less complex data, which may contain skewness and noisy data. Manual predictions may be inaccurate and time-consuming. This project aims to develop machine learning models to predict selling prices and lead statuses, addressing challenges such as data normalization, outlier detection, and leveraging regression and classification algorithms.

## Solution Approach

### Exploratory Data Analysis (EDA)
- Identify variable types and distributions.
- Treat 'Material_Reference' rubbish values starting with '00000'.
- Treat reference columns as categorical variables.
- Remove unnecessary 'INDEX'.

### Data Preprocessing
- Handle missing values using mean/median/mode.
- Treat outliers using IQR or Isolation Forest.
- Identify and treat skewness using appropriate transformations.
- Encode categorical variables using suitable techniques (one-hot encoding, label encoding, etc.).

### Feature Engineering
- Engineer new features if applicable.
- Drop highly correlated columns using a heatmap.

### Model Building and Evaluation
- Split the dataset into training and testing sets.
- Train and evaluate regression models for 'Selling_Price'.
- Train and evaluate classification models for 'Status' (WON/LOST).
- Optimize model hyperparameters using cross-validation and grid search.
- Interpret model results and assess performance.

### Model GUI (Streamlit)
- Create an interactive page.
- Input task (Regression/Classification) and column values.
- Perform feature engineering, scaling, and transformation steps.
- Predict new data and display the output.

### Tips
- Use the pickle module to dump and load models.
- Fit and transform in separate lines, use transform only for unseen data.

## Learning Outcomes

- Proficiency in Python and data analysis libraries (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Streamlit).
- Experience in data preprocessing techniques.
- Understanding and visualization of data through EDA techniques.
- Application of regression and classification machine learning techniques.
- Building and optimizing machine learning models.
- Feature engineering skills.
- Web application development using Streamlit.
- Understanding challenges and best practices in the manufacturing domain.

## Project Evaluation Metrics

- Modular code (functional blocks).
- Maintainability and portability.
- GitHub repository maintenance (public).
- Proper README file with project development details.
- Code adheres to PEP 8 coding standards.
- Demo video posted on LinkedIn.

## Execution Workflow

1. Clone the GitHub repository.
2. Set up the Python environment with required libraries.
3. Execute modular code blocks for data preprocessing, EDA, model building, and Streamlit app.
4. Follow coding standards mentioned in PEP 8.
5. Maintain GitHub repository with proper documentation.
6. Create a demo video showcasing the working model.
7. Post the demo video on LinkedIn.

**Note:** This README provides an overview of the project, its approach, learning outcomes, and evaluation metrics. For detailed instructions, refer to the codebase and documentation in the GitHub repository.
