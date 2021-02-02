# -*- coding: utf-8 -*-

import requests
import json
import pickle

url = 'http://127.0.0.1:5000/api/'

with open('df_client_dash.pkl','rb') as fp:
    df_client= pickle.load(fp)

with open('select_features.pkl','rb') as fp:
    select_features= pickle.load(fp)


data = data = df_client[select_features].loc[df_client.SK_ID_CURR == 381675]
#converting it into dictionary
data=data.to_dict('records')
#converting json
data_json = json.dumps(data)
headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
r = requests.post(url, data=data_json, headers=headers)
print(r, r.text)