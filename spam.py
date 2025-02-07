import pandas as pd  
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset (change 'spam.csv' to your actual file name)
df = pd.read_csv('spam.csv', encoding='latin-1')  

#only two columns is required   
df=df[['v1','v2']]
#naming the two columns
df.columns=['label','message']
#convert spam or not in machine understanding format by 0 and 1
df['label']=df['label'].map({'spam':1,'ham':0})
#to convert all string into lower case
df['message']=df['message'].str.lower()
#to remove punctuations and extra space
def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s', ' ' , text).strip()
    return text
df['message'] = df['message'].apply(clean_text)
#print(df.head())

#to convert text in numeric format
vectorizer=TfidfVectorizer()
x=vectorizer.fit_transform(df['message'])
#print(x.shape)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Define features (X) and target labels (y)
X = vectorizer.fit_transform(df['message'])  # Already transformed
y = df['label']  # Convert labels to binary (0 = ham, 1 = spam)

df = df.dropna(subset=['message'])  #run if its showing column value NaN and to remove it
#print(df.columns)
# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LogisticRegression(C=2)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

