import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle

def trainingModel(data_path="./data/Iris.csv", model_path="./model/iris_classifier.pkl"):
    """
    Train a DecisionTreeClassifier on the given dataset and save the model.
    
    Parameters:
    - data_path (str): Path to the dataset CSV file.
    - model_path (str): Path to save the trained model as a pickle file.

    Returns:
    - None
    """
    # Load the dataset
    df = pd.read_csv(data_path)
    
    # Define features and target
    X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    Y = df['Species']
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, shuffle=True)
    
    # Initialize and train the classifier
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    
    # Test the classifier and calculate accuracy
    y_pred = clf.predict(X_test)
    print(f"The accuracy of the model is {accuracy_score(y_test, y_pred) * 100:.2f}%")
    
    # Save the trained model
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    print(f"Model saved to {model_path}")
