# 🥗 DietMetrics – AI-Powered Nutri-Score and NOVA Classification

**Empowering healthier food decisions using Machine Learning and Natural Language Processing.**

## 🧠 About the Project

**DietMetrics** is an AI-driven web application designed to simplify complex food label data for everyday consumers. It automatically predicts the **Nutri-Score** and classifies the **NOVA food processing level** of any food product using advanced machine learning and NLP models.

 **Nutri-Score** is predicted using a custom-trained PyTorch regression model.  
 **NOVA Classification** is performed using a fine-tuned DistilBERT transformer.  
 All predictions are delivered through a lightweight Flask-based interface.

## 🔍 Features

-  Real-time prediction of Nutri-Score (-15 to +40) using nutritional values.
-  Ingredient-based NOVA classification (Level 1 to 4).
-  Supports manual input for nutrition values or ingredient lists.
-  Accurate predictions using models trained on **OpenFoodFacts** and **GroceryDB**.
-  Deployed via Flask backend with interactive UI.

## 📊 Technologies Used

- **Frontend:** HTML5, CSS3, Bootstrap  
- **Backend:** Flask (Python)  
- **ML Models:**  
  - PyTorch for Nutri-Score regression  
  - HuggingFace Transformers (DistilBERT) for NOVA classification  
- **Libraries:** Pandas, NumPy, Scikit-learn, Torch  
- **Deployment:** Render / Localhost


