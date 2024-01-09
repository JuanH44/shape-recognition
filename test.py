from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


if __name__ == "__main__":
    df = pd.read_csv('zernike_moments_2.csv')

    # Split the data into features and labels
    X = df.drop('shape', axis=1)
    y = df['shape']

    # Split the data into training and testing sets, shuffling the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

    # Create the model
    model = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=7)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Classification report
    print(classification_report(y_test, y_pred))

    # Plot the confusion matrix


