# 🧬 Automatic Leukemia Image Detection

## 🧠 Problem Statement

Leukemia, specifically Acute Lymphoblastic Leukemia (ALL), is a serious blood cancer that requires early detection for effective treatment. This project uses deep learning to classify blood cell images as either:
- `ALL`: Leukemic cells
- `HEM`: Healthy/normal cells

Our goal is to build an automated diagnostic tool that can support early leukemia detection from microscopic images.

---

## 📂 Dataset

- 📌 **Source**: [Kaggle - Leukemia Classification Dataset](https://www.kaggle.com/datasets/andrewmvd/leukemia-classification)
- 💡 **Contents**:
  - Labeled cell images: `ALL` (leukemia) and `HEM` (healthy)
  - Divided into training, validation, and test sets
- 🔄 Format: RGB image files

---

## ⚙️ Project Workflow

1. **Dataset Import and Preprocessing**:
   - Downloaded via Kaggle CLI
   - Resized images to 200x200
   - Optionally converted to grayscale
   - Checked class balance and image counts

2. **Exploratory Data Analysis (EDA)**:
   - Visualized distribution of classes
   - Displayed sample images for visual understanding

3. **Model Building**:
   - Baseline CNN model built using Keras
   - Configurable parameters: learning rate, batch size, epochs
   - Option to run on reduced dataset for low memory environments

4. **Training & Evaluation**:
   - Accuracy, loss curves plotted
   - Confusion matrix and classification report for validation results

5. **Result Visualization**:
   - Bar plots of class distribution
   - Performance metrics visualized
   - Saved trained model for future use

---

## 📈 Results

The project evaluated several classification models on leukemia image data. Below is a comparison of model performance on the test set:

| Model                       | Accuracy | Balanced Accuracy | ROC AUC | F1 Score | Time Taken (s) |
|----------------------------|----------|-------------------|---------|----------|----------------|
| SVC                        | **0.88** | **0.82**          | **0.82**| **0.87** | 94.63          |
| XGBClassifier              | 0.86     | 0.81              | 0.81    | 0.86     | 292.92         |
| Linear Discriminant Analysis | 0.85  | 0.81              | 0.81    | 0.85     | 18.73          |
| RidgeClassifierCV          | 0.85     | 0.81              | 0.81    | 0.85     | 15.88          |
| RidgeClassifier            | 0.85     | 0.81              | 0.81    | 0.85     | **1.46**       |

🔍 **Best Overall Model**:  
- **SVC** achieved the highest accuracy (**88%**) and F1 score (**0.87**), making it the top-performing model in terms of classification quality.

🕒 **Fastest Model**:  
- **RidgeClassifier** trained in just **1.46 seconds** while still achieving a strong F1 score (**0.85**), making it a great choice for speed-critical applications.

---

## ✅ Conclusion

This project demonstrates how deep learning can effectively classify leukemia from blood smear images. It shows potential for real-world use in supporting hematologists and early cancer detection efforts.


## ▶️ How to Explore the Project

To run the notebook:

1. Open the `.ipynb` file in Google Colab or Jupyter Notebook
2. Ensure you upload your `kaggle.json` and run the first setup cell
3. Execute all cells to train the model and view results

---

## 🧰 Requirements

To recreate the environment:

```bash
tensorflow
numpy
pandas
matplotlib
seaborn
opencv-python
scikit-learn
