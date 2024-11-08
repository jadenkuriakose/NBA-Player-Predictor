from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_executor import Executor
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing
executor = Executor(app)  # To run background tasks

# Function to get last N games of data for a player
def get_last_n_games(player_name, season='2023', n_games=15):
    name_parts = player_name.split()
    if len(name_parts) == 2:
        first_name, last_name = name_parts
    elif len(name_parts) == 1:
        first_name, last_name = name_parts[0], name_parts[0]
    else:
        return {"error": "Invalid player name format. Please provide a valid NBA player's name."}

    # Construct player ID as used by Basketball Reference
    player_id = f"{last_name[:5]}{first_name[:2]}01".lower()
    url = f"https://www.basketball-reference.com/players/{last_name[0].lower()}/{player_id}/gamelog/{season}"

    response = requests.get(url)
    if response.status_code != 200:
        return {"error": "Could not retrieve data. Check player name and season."}

    soup = BeautifulSoup(response.text, 'html.parser')
    stats_table = soup.find('table', {'id': 'pgl_basic'})

    if not stats_table:
        return {"error": "No stats found for the player."}

    # Extract data into a DataFrame
    data = []
    for row in stats_table.find_all('tr', class_=lambda x: x != 'thead'):
        cols = [col.get_text() for col in row.find_all('td')]
        if cols:
            data.append(cols)
    
    columns = [th.get_text() for th in stats_table.find('thead').find_all('th')][1:]  # Skip index
    stats_df = pd.DataFrame(data, columns=columns)
    stats_df = stats_df.apply(pd.to_numeric, errors='ignore')

    # Select only the last N games
    stats_df = stats_df.tail(n_games).reset_index(drop=True)
    return stats_df

# Function to convert MP from "MM:SS" format to total minutes as float
def convert_minutes(mp):
    try:
        minutes, seconds = map(int, mp.split(":"))
        return minutes + seconds / 60
    except:
        return 0  # Handle any non-numeric values as zero playing time

# Function to preprocess data
def preprocess_data(stats_df):
    stats_df['MP'] = stats_df['MP'].apply(convert_minutes)

    features = ['PTS', 'AST', 'TRB', 'MP', 'FGA', '3PA', 'TOV', 'PF']
    X = stats_df[features].fillna(0)

    y = stats_df['PTS'].shift(-1).dropna()  # Target is the points scored in the next game

    X = X.iloc[:len(y)]  # Reduce X to match the length of y

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

# Function to train and evaluate the model using Random Forest
def train_model(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return model, mse

# Function to make a prediction for the next game
def predict_next_game(model, last_game_stats):
    predicted_points = model.predict(last_game_stats)
    return predicted_points[0]

# Background task for processing and prediction
def process_and_predict(player_name):
    # Get the last N games
    stats_df = get_last_n_games(player_name, season='2023', n_games=15)
    if isinstance(stats_df, dict) and 'error' in stats_df:
        return stats_df
    
    # Preprocess the data
    X, y = preprocess_data(stats_df)

    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train the model
    model, mse = train_model(X_train, y_train, X_test, y_test)

    # Predict the next game
    last_game = X[-1].reshape(1, -1)  # Use the last game's stats
    predicted_points = predict_next_game(model, last_game)

    return {"predicted_points": predicted_points, "mse": mse}

# API endpoint to get the prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    player_name = data.get('player_name')

    if not player_name:
        return jsonify({"error": "Player name is required"}), 400

    # Run the background task
    future = executor.submit(process_and_predict, player_name)

    # Wait for the task to complete and get the result
    result = future.result()

    if 'error' in result:
        return jsonify(result), 400

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port = 8080)
