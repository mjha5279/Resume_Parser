import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle
import re

# ---------------------------
# 1. Load dataset
# ---------------------------
df = pd.read_csv("UpdatedResumeDataSet.csv")  # Make sure this CSV exists in the same folder
# Assume it has columns: 'Resume' (text) and 'Category' (job role)
print(df.head())

# ---------------------------
# 2. Clean resume text
# ---------------------------
def clean_resume(text):
    text = re.sub(r'http\S+\s', ' ', text)
    text = re.sub(r'RT|cc', ' ', text)
    text = re.sub(r'#\S+\s', ' ', text)
    text = re.sub(r'@\S+', '  ', text)
    text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

df['Cleaned_Resume'] = df['Resume'].apply(clean_resume)

# ---------------------------
# 3. Encode labels
# ---------------------------
le = LabelEncoder()
y = le.fit_transform(df['Category'])
pickle.dump(le, open("encoder.pkl", "wb"))
print("Saved encoder.pkl")

# ---------------------------
# 4. Vectorize text with TF-IDF
# ---------------------------
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['Cleaned_Resume'])
pickle.dump(tfidf, open("tfidf.pkl", "wb"))
print("Saved tfidf.pkl")

# ---------------------------
# 5. Train classifier
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svc_model = SVC(kernel='linear', probability=True)
svc_model.fit(X_train, y_train)

# Evaluate (optional)
accuracy = svc_model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# ---------------------------
# 6. Save trained model
# ---------------------------
pickle.dump(svc_model, open("clf.pkl", "wb"))
print("Saved clf.pkl")

print("All files ready: clf.pkl, tfidf.pkl, encoder.pkl")
