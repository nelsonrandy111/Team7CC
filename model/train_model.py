from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd

url = 'https://raw.githubusercontent.com/nelsonrandy111/Team7CC/main/data/smote_balanced_symbipredict_2022.csv'
df = pd.read_csv(url)

X = df.drop(columns=['prognosis', 'family_history'])
y = df['prognosis']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)

steps=[('lg',LogisticRegression())]
pipe=Pipeline(steps)

pipe.fit(X_train, y_train)

joblib.dump(pipe, 'prognosis_model.pkl')

print("Model training complete and saved to 'prognosis_model.pkl'")