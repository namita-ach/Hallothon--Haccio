import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from flask import Flask, request, render_template
from sklearn import metrics

# Load your data here
data = pd.read_csv(r"C:\Users\VivekM\FINAL YAY.csv")

x = data[['Topic', 'Mode of Communication']]
y = data['Effectiveness']

# Split data into training and testing, and train a linear regression model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.40, shuffle=True)
model = LinearRegression()
model.fit(x_train, y_train)

# Your OnlineLinearRegression class
class OnlineLinearRegression:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights = np.zeros(2)

    def update(self, x, y):
        error = y - self.predict(x)
        self.weights += self.learning_rate * error * x

    def predict(self, x):
        return np.dot(self.weights, x)

app = Flask(__name__)
online_model = OnlineLinearRegression()

@app.route('/')
def index():
    return render_template('student_form.html')

@app.route('/student_form', methods=['POST'])
def update_online_model():
    topic = int(request.form['topic'])
    mode_of_teaching = int(request.form['mode_of_teaching'])
    effectiveness = int(request.form['effectiveness'])

    # Assuming your OnlineLinearRegression accepts 'topic' and 'mode_of_teaching' as x and 'effectiveness' as y
    x = np.array([topic, mode_of_teaching])
    y = effectiveness
    online_model.update(x, y)
    
    return 'Model Updated'

if __name__ == '__main__':
    app.run(debug=True)
