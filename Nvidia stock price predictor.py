import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

# Read the stock price dataset
data = pd.read_csv("C:\\Python\\Projects\\Nvidia Stock Price Predictor\\Nvidia 1 year stock outcome.csv")

# Remove commas from 'Volume' column and convert to numeric
data['Volume'] = data['Volume'].str.replace(',', '').astype(float)

# Feature Engineering: Create a 7-day moving average
data['7_day_MA'] = data['Close'].rolling(window=7).mean()

# Drop rows with NaN values created by the rolling window
data = data.dropna()

# Define the features and target variable
features = ['Open', 'Volume', 'High', 'Low', '7_day_MA']
target = 'Close'

# Split the data into training (80%) and testing (20%) datasets
train_data = data.iloc[:int(.80 * len(data)), :]
test_data = data.iloc[int(.80 * len(data)):, :]

# Create and train the XGBoost regression model with hyperparameter tuning
model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5)
model.fit(train_data[features], train_data[target])

# Make predictions on the test data
predictions = model.predict(test_data[features])

# Calculate model accuracy (R² score)
accuracy = model.score(test_data[features], test_data[target])
print('Model Accuracy (R² score):', accuracy)

# Plot the predictions vs actual close prices for the test data
plt.plot(test_data[target].index, predictions, label='Predicted Close Price')
plt.plot(test_data[target].index, test_data[target], label='Actual Close Price')

# Adding labels and title
plt.xlabel('Index')
plt.ylabel('Close Price')
plt.title('Predicted vs Actual Close Price')
plt.legend()
plt.show()
