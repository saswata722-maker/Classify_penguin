import pandas as pd 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import classification_report , accuracy_score 

df = sns.load_dataset('penguins')

x=df.drop(columns = ['species','island'])
y=df['species']

num_cols = x.select_dtypes(include=['float64','int64']).columns
imputer = SimpleImputer(strategy = 'mean')
x[num_cols] = imputer.fit_transform(x[num_cols])
x['sex'] = x['sex'].fillna(x['sex'].mode()[0])

x = pd.get_dummies(x,columns =['sex'], drop_first = True)
le = LabelEncoder()
y = le.fit_transform(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.2,random_state = 42 , stratify = y)

model = LogisticRegression(max_iter =  1000 )
model.fit(x_train,y_train)

def predict_penguin(bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex):
    new_data = pd.DataFrame([{
        'bill_length_mm': bill_length_mm,
        'bill_depth_mm': bill_depth_mm,
        'flipper_length_mm': flipper_length_mm,
        'body_mass_g': body_mass_g,
        'sex': sex
    }])
    new_data = pd.get_dummies(new_data, columns=['sex'], drop_first=True)
    new_data = new_data.reindex(columns=x_train.columns, fill_value=0) 
    pred = model.predict(new_data)
    return le.inverse_transform(pred)[0]

print("Enter penguin measurements:")
bl = float(input("Bill length (mm): "))
bd = float(input("Bill depth (mm): "))
fl = float(input("Flipper length (mm): "))
bm = float(input("Body mass (g): "))
sx = input("Sex (Male/Female): ")

species = predict_penguin(bl, bd, fl, bm, sx)
print(f"Predicted species: {species}")
y_pred = model.predict(x_test)

#print(f"Accuracy{accuracy_score(y_test,y_pred):.2f}")
#print(classification_report(y_test,y_pred,target_names=le.classes_))