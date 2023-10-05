import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# Load the data
data = pd.read_csv('data.csv')
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'], test_size=0.25, random_state=0)
# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)
# Evaluate the model
score = model.score(X_test, y_test)
print('Accuracy:', score)
# Save the model
model.save('model.pkl')