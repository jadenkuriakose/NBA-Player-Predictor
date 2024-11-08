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

app = Flask(__name__)
CORS(app)
executor = Executor(app)

def find_player_page(first_name, last_name):
    base = "https://www.basketball-reference.com"
    for number in range(1, 7):
        player_id = f"{last_name[0].lower()}/{last_name.lower()[:5]}{first_name.lower()[:2]}0{number}.html"
        url = f"{base}/players/{player_id}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            player_header = soup.find('h1')
            if player_header:
                player_name = player_header.text.strip()
                if player_name.lower() == f"{first_name} {last_name}".lower():
                    return url
        except requests.RequestException as e:
            print(f"An error occurred while fetching stats for {url}: {e}")
    print(f"No matching player found for {first_name} {last_name} after 6 attempts.")
    return None

def get_last_n_games(player_name, season='2023', n_games=15):
    name_parts = player_name.split()
    if len(name_parts) == 2:
        first_name, last_name = name_parts
    elif len(name_parts) == 1:
        first_name, last_name = name_parts[0], name_parts[0]
    else:
        return {"error": "Invalid player name format. Please provide a valid NBA player's name."}
    player_url = find_player_page(first_name, last_name)
    if not player_url:
        return {"error": f"Could not find player page for {first_name} {last_name}."}
    player_id = player_url.split('/')[-1].split('.')[0]
    url = f"https://www.basketball-reference.com/players/{player_id[:1]}/{player_id}/gamelog/{season}"
    response = requests.get(url)
    if response.status_code != 200:
        return {"error": "Could not retrieve game log data. Check player name and season."}
    soup = BeautifulSoup(response.text, 'html.parser')
    stats_table = soup.find('table', {'id': 'pgl_basic'})
    if not stats_table:
        return {"error": "No game log found for the player."}
    data = []
    for row in stats_table.find_all('tr')[1:]:
        cols = row.find_all('td')
        if len(cols) > 0:
            game_data = [col.get_text() for col in cols]
            data.append(game_data)
    columns = [th.get_text() for th in stats_table.find('thead').find_all('th')][1:]
    stats_df = pd.DataFrame(data, columns=columns)
    stats_df = stats_df.apply(pd.to_numeric, errors='ignore')
    stats_df = stats_df.tail(n_games).reset_index(drop=True)
    return stats_df

def convert_minutes(mp):
    try:
        minutes, seconds = map(int, mp.split(":"))
        return minutes + seconds / 60
    except:
        return 0

def preprocess_data(stats_df):
    stats_df['MP'] = stats_df['MP'].apply(convert_minutes)
    features = ['PTS', 'AST', 'TRB', 'MP', 'FGA', '3PA', 'TOV', 'PF']
    X = stats_df[features].fillna(0)
    y = stats_df['PTS'].shift(-1).dropna()
    X = X.iloc[:len(y)]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def train_model(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return model, mse

def predict_next_game(model, last_game_stats):
    predicted_points = model.predict(last_game_stats)
    return predicted_points[0]

def process_and_predict(player_name):
    stats_df = get_last_n_games(player_name, season='2023', n_games=15)
    if isinstance(stats_df, dict) and 'error' in stats_df:
        return stats_df
    X, y = preprocess_data(stats_df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model, mse = train_model(X_train, y_train, X_test, y_test)
    last_game = X[-1].reshape(1, -1)
    predicted_points = predict_next_game(model, last_game)
    return {"predicted_points": predicted_points, "mse": mse}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    player_name = data.get('player_name')
    if not player_name:
        return jsonify({"error": "Player name is required"}), 400
    future = executor.submit(process_and_predict, player_name)
    result = future.result()
    if 'error' in result:
        return jsonify(result), 400
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=8080)
