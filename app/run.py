# Set up workspace
import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

# Tokenize
def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# Load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('InsertTableName',engine)

# Load model
model = joblib.load("../models/classifier.pkl")


# Index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # Extract data needed for visuals
    gen_count = df.groupby('genre').count()['message']
    gen_per = round(100*gen_count/gen_count.sum(), 2)
    gen = list(gen_count.index)
    cat_num = df.drop(['id', 'message', 'original', 'genre'], axis = 1).sum()
    cat_num = cat_num.sort_values(ascending = False)
    cat = list(cat_num.index)
    
    colors = ['yellow', 'green', 'red']
    
    # Create visuals
    graphs = [
        {
            "data": [
              {
                "type": "bar",
                "x": cat,
                "y": cat_num,
                "marker": {
                  "color": '7D3C98'}
                }
            ],
            "layout": {
              "title": "Number of Messages by Category",
              'yaxis': {
                  'title': "Count"
              },
              'xaxis': {
                  'title': "Category"
              },
              'barmode': 'group'
            }
        },                                                                                                         
        {
            "data": [
              {
                "type": "pie",
                "uid": "f4de1f",
                "hole": 0.4,
                "name": "Source",
                "pull": 0,
                "domain": {
                  "x": gen_per,
                  "y": gen
                },
                "marker": {
                  "colors": [
                    "#CB4335",
                    "#2E86C1",
                    "#16A085"
                   ]
                },
                "textinfo": "label+value",
                "hoverinfo": "all",
                "labels": gen,
                "values": gen_count
              }
            ],
            "layout": {
              "title": "Percent of Messages by Source"
            }
        }
    ]

    
    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# Web page that handles user query and displays model results
@app.route('/go')
def go():
    # Save user input in query
    query = request.args.get('query', '') 

    # Use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()