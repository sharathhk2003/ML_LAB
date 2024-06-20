import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('./daily_weather.csv')
data.drop(columns=['number'], inplace=True)
data.dropna(inplace=True)

clean_data = data.copy()
clean_data['high_humidity_label'] = (clean_data['relative_humidity_3pm'] > 24.99).astype(int)

morning_features = [
    'air_pressure_9am', 'air_temp_9am', 'avg_wind_direction_9am',
    'avg_wind_speed_9am', 'max_wind_direction_9am', 'max_wind_speed_9am',
    'rain_accumulation_9am', 'rain_duration_9am', 'relative_humidity_9am'
]

X = clean_data[morning_features]
y = clean_data['high_humidity_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=324)

humidity_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
humidity_classifier.fit(X_train, y_train)

y_predicted = humidity_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_predicted) * 100

print(f"Accuracy: {accuracy:.2f}%")
