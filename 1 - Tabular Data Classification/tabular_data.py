""" Tabular Data Classification with AutoML """

# Tabular Data Classification with AutoML

# Import libraries
# Pandas is used for data manipulation
import pandas as pd

# TensorFlow and AutoKeras are used for AutoML
import tensorflow as tf

# Import AutoKeras library as ak to use the AutoML functionality for structured data classification tasks
import autokeras as ak

# Train test split is used to split the data into training and test sets
from sklearn.model_selection import train_test_split

# Set logging level to hide warnings
tf.get_logger().setLevel("ERROR")

# Set folder path
FILE_PATH = "C:/Demos/1 - Tabular Data Classification/Titanic.csv"

# Read the data into a data frame
data = pd.read_csv(FILE_PATH)

# Inspect the data set
data.head()
len(data)

# Split the data into training and test sets
train_x, test_x = train_test_split(data, test_size=0.2)

# Inspect the size of the data sets
len(train_x)
len(test_x)

# Pop the last column off the data frame
train_y = train_x.pop("Survived")
test_y = test_x.pop("Survived")

# Inspect the training data set
train_x.head()
train_y.head()

# Inspect the test data set
test_x.head()
test_y.head()

# Create a structured data classifier
classifier = ak.StructuredDataClassifier(max_trials=5)

# Train the model
classifier.fit(x=train_x, y=train_y, epochs=10)

# Summarize the best model
classifier.export_model().summary()

# Evaluate the model
score = classifier.evaluate(x=test_x, y=test_y)

# Inspect the accuracy
print(score[1])

# Will Rose survive the Titanic?
rose_x = pd.DataFrame(
    data=[
        {
            "Sex": "female",
            "Age": 17,
            "Family": 2,
            "Class": 1,
            "Fare": 150.0,
            "Cabin": "B",
            "Port": "Southampton",
        }
    ]
)

# Inspect Rose's data
rose_x.head()

# Predict if Rose will survive
classifier.predict(rose_x)[0][0]

# Will Jack survive?
jack_x = pd.DataFrame(
    data=[
        {
            "Sex": "male",
            "Age": 20,
            "Family": 0,
            "Class": 3,
            "Fare": 15.0,
            "Cabin": "F",
            "Port": "Southampton",
        }
    ]
)

# Predict if Jack will survive
classifier.predict(jack_x)[0][0]
