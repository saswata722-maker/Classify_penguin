# 🐧 Palmer Penguins Species Classifier

A machine learning project built with Python and Scikit-Learn that predicts the species of a penguin (*Adelie*, *Chinstrap*, or *Gentoo*) based on physical characteristics (bill length, bill depth, flipper length, body mass, and sex).

---

## 📌 Features

- **Dataset Preprocessing**: Handles missing values with mean imputation (`SimpleImputer`) and mode imputation for categorical data.
- **Feature Encoding**: Utilizes one-hot encoding (`pd.get_dummies`) and label encoding (`LabelEncoder`).
- **Logistic Regression Model**: Trained on the Palmer Penguins dataset with stratified train/test split.
- **Interactive Prediction**: CLI-based interactive prompt allowing users to enter custom measurements and receive real-time species predictions.

---

## 📊 Dataset Overview

The model uses the **Palmer Penguins** dataset (via Seaborn), which includes morphological measurements for three penguin species collected from islands in the Palmer Archipelago, Antarctica:

- `bill_length_mm`: Bill length in millimeters
- `bill_depth_mm`: Bill depth in millimeters
- `flipper_length_mm`: Flipper length in millimeters
- `body_mass_g`: Body mass in grams
- `sex`: Biological sex (Male/Female)

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/penguin-classifier.git
cd penguin-classifier
```

### 2. Set Up a Virtual Environment (Optional but Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Classifier

```bash
python penguin.py
```

---

## 💻 Example Usage

```text
Enter penguin measurements:
Bill length (mm): 39.1
Bill depth (mm): 18.7
Flipper length (mm): 181
Body mass (g): 3750
Sex (Male/Female): Male

Predicted species: Adelie
```

---

## 📁 Project Structure

```text
penguin/
├── .gitignore          # Git ignore rules
├── requirements.txt    # Project dependencies
├── README.md           # Project documentation
└── penguin.py          # Training and prediction script
```

---

## 🛠️ Built With

- [Python](https://www.python.org/)
- [Scikit-Learn](https://scikit-learn.org/)
- [Pandas](https://pandas.pydata.org/)
- [Seaborn](https://seaborn.pydata.org/)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
