# Business Machine Learning Projects

This repository contains five end-to-end machine learning projects, each focused on a different domain and addressing various business problems. Each project includes a Jupyter notebook that walks through the entire process, from data preprocessing and feature engineering to model training, evaluation, and deployment.

## Project List

1. **Churn Prediction**

   - **Notebook:** [Churn_Prediction.ipynb](./Churn_Prediction.ipynb)
   - **Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pt9wsBjFOCoHRrBPXLuvW4glYaw1tZh5?usp=sharing)
   - **Description:** This project focuses on predicting customer churn for a telecommunications company using the Telco Customer Churn dataset. The goal is to identify customers who are likely to leave the service so that proactive retention strategies can be implemented.
   - **Key Techniques:** Data preprocessing, feature engineering, Random Forest, SMOTE, hyperparameter tuning.

2. **Credit Card Fraud Detection**

   - **Notebook:** [CreditCard_Fraud_Detection.ipynb](./CreditCard_Fraud_Detection.ipynb)
   - **Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fFeSKfIeoxdjW4eU1ZKltQHwrumfzIRl?usp=sharing)
   - **Description:** This project involves building a model to detect fraudulent transactions in a credit card dataset. The model helps in identifying suspicious activities and preventing financial losses.
   - **Key Techniques:** Data balancing using SMOTE, feature scaling, Random Forest, XGBoost, ROC-AUC evaluation.

3. **Recommendation System**

   - **Notebook:** [Recommendation_System.ipynb](./Recommendation_System.ipynb)
   - **Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GHP3Z-8SbMlJekCz099FFArjlFp5ryjL?usp=sharing)
   - **Description:** This project implements a recommendation system using collaborative filtering on the MovieLens dataset. The system suggests movies to users based on their viewing history and preferences.
   - **Key Techniques:** Collaborative filtering, SVD, cross-validation, top-N recommendations.

4. **Sales Forecasting**

   - **Notebook:** [Sales_Forecasting.ipynb](./Sales_Forecasting.ipynb)
   - **Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oCtjodwXu9ExC3jb5Y0RjnRImGx0YPoL?usp=sharing)
   - **Description:** This project focuses on predicting future sales for Rossmann stores using historical sales data. Accurate sales forecasting helps in inventory management, resource allocation, and financial planning.
   - **Key Techniques:** Time series forecasting, feature engineering, XGBoost, RMSE evaluation, hyperparameter tuning.

5. **Customer Segmentation**
   - **Notebook:** [Customer_Segmentation.ipynb](./Customer_Segmentation.ipynb)
   - **Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_8bdZeed89U2xzqJElSN43vmb5uii_1M?usp=sharing)
   - **Description:** In this project, customer data from a mall is used to segment customers into distinct groups based on their behavior and spending patterns. This segmentation can be used to tailor marketing strategies and improve customer engagement.
   - **Key Techniques:** K-Means clustering, feature scaling, Elbow method, Silhouette score, cluster visualization.

## Requirements

To run these projects, you need to have the following libraries installed:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `xgboost`
- `imbalanced-learn`

You can install the required packages using the following command:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost imbalanced-learn
```

## How to Run the Notebooks

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/pavan98765/Business_projects.git
   ```

2. Navigate to the project directory:

   ```bash
   cd machine-learning-projects
   ```

3. Open any of the Jupyter notebooks:
   ```bash
   jupyter notebook Churn_Prediction.ipynb
   ```
4. Follow the instructions in each notebook to run the code and reproduce the results.

## Acknowledgments

The datasets used in these projects are sourced from public datasets available on platforms like Kaggle and UCI Machine Learning Repository.
Special thanks to the open-source community for providing tools and resources that made these projects possible.
