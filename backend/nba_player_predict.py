from flask import Flask, request, jsonify
from flask_cors import CORS 
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import unicodedata
import html



app = Flask(__name__)
CORS(app)


def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return ''.join([c for c in nfkd_form if not unicodedata.combining(c)])

def find_player_page(first_name, last_name):
    base = "https://www.basketball-reference.com"
    for number in range(1, 7):
        player_id = f"{last_name[0].lower()}/{last_name.lower()[:5]}{first_name.lower()[:2]}0{number}.html"
        url = f"{base}/players/{player_id}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            response.encoding = 'utf-8'
            
            soup = BeautifulSoup(response.text, 'html.parser')

            player_header = soup.find('h1')
            if player_header:
                player_name = player_header.text.strip()

                player_name = html.unescape(player_name)

                normalized_name = remove_accents(player_name)

                if normalized_name.lower() == f"{first_name} {last_name}".lower():
                    return url
        except requests.RequestException as e:
            print(f"An error occurred while fetching stats for {url}: {e}")
    
    print(f"No matching player found for {first_name} {last_name} after 6 attempts.")
    return None

def get_last_n_games(player_url, season='2024', n_games=30):
    response = requests.get(player_url.replace('.html', f'/gamelog/{season}'))
    if response.status_code != 200:
        print("Could not retrieve data. Check player URL and season.")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')
    stats_table = soup.find('table', {'id': 'pgl_basic'})

    if not stats_table:
        print("No stats found for the player.")
        return None

    data = []
    for row in stats_table.find_all('tr', class_=lambda x: x != 'thead'):
        cols = [col.get_text() for col in row.find_all('td')]
        if cols:
            data.append(cols)
    
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
    features = ['MP', 'FGA', '3PA', 'FTA', 'TOV', 'PF']
    targets = ['PTS', 'AST', 'TRB']
    
    X = stats_df[features].fillna(0)
    
    y_dict = {}
    for target in targets:
        y_dict[target] = stats_df[target].shift(-1).dropna()
    
    X = X.iloc[:len(y_dict[targets[0]])]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y_dict, features, targets

class MultiStatPredictor:
    def __init__(self):
        self.models = {}
        self.features = None
        self.targets = None
    
    def train(self, X_train, y_dict_train, X_test, y_dict_test):
        self.models = {}
        results = {}
        
        for target in y_dict_train.keys():
            model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            model.fit(X_train, y_dict_train[target])
            
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_dict_test[target], predictions)
            rmse = mse ** 0.5
            
            self.models[target] = model
            results[target] = rmse
        
        return results
    
    def predict_next_game(self, last_game_stats):
        predictions = {}
        for target, model in self.models.items():
            predicted_value = model.predict(last_game_stats)[0]
            predictions[target] = round(predicted_value, 1)
        return predictions

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    player_name = data.get('player_name')
    season = '2024'

    name_parts = player_name.split()
    if len(name_parts) != 2:
        return jsonify({'error': 'Invalid player name format. Please provide first and last name.'}), 400
    first_name, last_name = name_parts

    # Use find_player_page to get the player URL
    player_url = find_player_page(first_name, last_name)
    if player_url is None:
        return jsonify({'error': 'Player not found.'}), 404

    stats_df = get_last_n_games(player_url, season=season, n_games=30)
    if stats_df is None:
        return jsonify({'error': 'Player stats not found.'}), 404

    X, y_dict, features, targets = preprocess_data(stats_df)

    X_train, X_test = train_test_split(X, test_size=0.2, shuffle=False)
    
    y_dict_train = {}
    y_dict_test = {}
    for target in targets:
        y_train, y_test = train_test_split(y_dict[target], test_size=0.2, shuffle=False)
        y_dict_train[target] = y_train
        y_dict_test[target] = y_test

    predictor = MultiStatPredictor()
    predictor.train(X_train, y_dict_train, X_test, y_dict_test)

    last_game = X[-1].reshape(1, -1)
    predictions = predictor.predict_next_game(last_game)
    
    return jsonify(predictions)

if __name__ == "__main__":
    app.run(debug=True, port=8080)
