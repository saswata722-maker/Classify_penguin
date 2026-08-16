import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Load dataset
df = sns.load_dataset('penguins')

# 2. Features and target separation
x = df.drop(columns=['species', 'island'])
y = df['species']

# 3. Impute missing numerical values with column mean
num_cols = x.select_dtypes(include=['float64', 'int64']).columns
imputer = SimpleImputer(strategy='mean')
x[num_cols] = imputer.fit_transform(x[num_cols])

# 4. Impute missing categorical values with mode
x['sex'] = x['sex'].fillna(x['sex'].mode()[0])

# 5. One-hot encode categorical features and label encode target
x = pd.get_dummies(x, columns=['sex'], drop_first=True)
le = LabelEncoder()
y = le.fit_transform(y)

# 6. Train-test split (80/20) with stratification
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)


def predict_penguin(bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex):
    """Predict the penguin species from physical measurements and sex."""
    # Standardize sex input format (e.g., 'male' -> 'Male')
    formatted_sex = sex.strip().capitalize() if isinstance(sex, str) else sex

    new_data = pd.DataFrame([{
        'bill_length_mm': bill_length_mm,
        'bill_depth_mm': bill_depth_mm,
        'flipper_length_mm': flipper_length_mm,
        'body_mass_g': body_mass_g,
        'sex': formatted_sex
    }])
    new_data = pd.get_dummies(new_data, columns=['sex'], drop_first=True)
    new_data = new_data.reindex(columns=x_train.columns, fill_value=0)
    pred = model.predict(new_data)
    return le.inverse_transform(pred)[0]


if __name__ == '__main__':
    # Model evaluation metrics
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained successfully. Test Accuracy: {accuracy * 100:.2f}%\n")

    print("Enter penguin measurements:")
    try:
        bl = float(input("Bill length (mm): "))
        bd = float(input("Bill depth (mm): "))
        fl = float(input("Flipper length (mm): "))
        bm = float(input("Body mass (g): "))
        sx = input("Sex (Male/Female): ")

        species = predict_penguin(bl, bd, fl, bm, sx)
        print(f"\n✨ Predicted species: {species}")
    except ValueError as e:
        print(f"Invalid input: {e}")