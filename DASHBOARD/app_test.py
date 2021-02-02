# Run this app with `python app.py` 

import shap
import numpy as np
import pandas as pd 
#import plotly.express as px
#from jupyter_dash import JupyterDash
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import _io as io
import base64
from sklearn.metrics import confusion_matrix
import pickle
#from shap import force_plot

with open('select_features.pkl','rb') as fp:
    selected_features = pickle.load(fp)

with open('colnum.pkl', 'rb') as f:
    colnum = pickle.load(f)  

with open('df_client_dash.pkl','rb') as fp:
    df_client= pickle.load(fp)


with open('encoded_logo.pkl','rb') as fp:
    encoded_logo= pickle.load(fp)   
    
with open('encoded_sum.pkl','rb') as fp:
    encoded_sum= pickle.load(fp)   
    
with open('lgbm.pkl','rb') as fp:
    lgbm= pickle.load(fp)   

with open('encoded_sum.pkl','rb') as fp:
    encoded_sum= pickle.load(fp)

def display_summary_plot(encoded_sum) : 
    #plt.close('all') #pour éviter les superposition de graphiques
    shap_html =''
    #summary_plot = shap.summary_plot(shap_values[1], df_client[selected_features],plot_type="violin", 
                                     #show=False,max_display=10)
    #color='RdGy',
    """ figure to html base64 png image """ 
    #tmpfile_sum = io.BytesIO()
    #plt.tight_layout()
    #plt.savefig(tmpfile_sum, format='png',facecolor='w',transparent=False,edgecolor=color,dpi=70)
    #encoded_sum = base64.b64encode(tmpfile_sum.getvalue()).decode('utf-8')
    shap_html = html.Img(src=f"data:image/png;base64, {encoded_sum}")
    #del tmpfile_sum
    #del summary_plot
    #del encoded_sum
    #plt.close('all')
    return shap_html


#definition de liste de variables pour les affichage
colpie=['CODE_GENDER','FLAG_OWN_CAR',
     'FLAG_OWN_REALTY','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE']
coldef = colpie + colnum
coldef.sort()

#couleur gris logo
color = '#D8D8D8'
#couleur bleue
bleue= '#1e88e5'
#couleur rouge
rouge = '#ff0d57'

external_stylesheets = ['https://cdn.rawgit.com/plotly/dash-app-stylesheets/2d266c578d2a6e8850ebce48fdb52759b2aef506/stylesheet-oil-and-gas.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.layout = html.Div(style={'backgroundColor': "color"},children=[
    

                    html.Img(src='data:image/png;base64,{}'.format(encoded_logo)),
                    html.H1(children='Tableau de bord "Prêt à depenser" '),
                    html.Label(['Choisissez un client:'],style={'font-weight': 'bold', "text-align": "left"}),
    
                    dcc.Dropdown(id='dropdown_client', options=[
                    {'label': i, 'value': i} for i in df_client.SK_ID_CURR.sort_values().unique()
                        ], multi=False, placeholder='Please select...',style=dict(width='50%')),
                    dcc.Graph(id='tableau'
                    ),
                    dcc.Markdown(id='display_loan',style={'marginLeft': 50 }),
                    html.H5(children='Probabilité de défaut de crédit',style={'color': rouge }),
                    
                    dcc.Graph(id='indicateur',style={'text-align':'center'}),
                                        dcc.Markdown('''  
**Seuil de défaut : 49.4 %**  
**Seuil au minimum de pertes attendues : 73%**  
*Taux de perte si défaut de crédit: 70%*      
*Taux de perte si refus de crédit : 20%* ''',style={'marginLeft': 50,'color': 'gray','margin-top':'10px','font-size': '12px'} ),

                    dcc.Markdown(id='disp_prob'),
                    html.Label(children='''
                       Variables importantes du modèle
                   ''',style={'marginLeft': 50,'text-align':'left'}),
                   html.Label(children='''
                           ROUGE: valeurs hautes de la variable                                                        
                                   ''',style={'marginLeft': 50,'text-align':'left','color': rouge,'font-size': '12px'}),
                 html.Label(children='''
                           BLEU: valeurs basses de la variable                                                       
                                   ''',style={'marginLeft': 50,'text-align':'left','color': bleue,'font-size': '12px'}),
                        #affichage du summer plot shap violon
 
                html.Div(children = display_summary_plot(encoded_sum),style={"border":"2px"}),
                html.Label(['Choisissez une valeur à afficher:'],style={ "text-align": "left"}),
                dcc.Dropdown(id='dropdown_num', options=[
                    {'label': i, 'value': i} for i in df_client[colnum].columns.sort_values()
                        ], multi=False, placeholder='Please select...'), 
                dcc.Checklist(id='check_num',
                                options=[
                            {'label': 'Par caratéristique', 'value': 'yes'},
                            ], ),  
                    
                dcc.Graph(id='bar_plot'),  
                
                html.Label(['Choisissez une caractéristique à explorer:'],style={ "text-align": "left"}),
                    dcc.Dropdown(id='dropdown_cat', options=[
                    {'label': i, 'value': i} for i in colpie
                        ], multi=False, placeholder='Please select...'), 
               dcc.RadioItems(
                                id='filter_pie',
                                options=[{'label': i, 'value': i} for i in ['Clients en défaut', 'solvables']],
                                value='solvables',
                                labelStyle={'display': 'inline-block'}),
               dcc.Graph(id='pie_plot'), 
                    
                     

])
   
#pie plot
@app.callback(Output('pie_plot', 'figure'), 
              Input('dropdown_client', 'value'),
              Input('dropdown_cat','value'),
              Input('filter_pie','value'))
def pie_plot(id_client, parameter,classe):
    
    if classe == 'solvables':
        y = 0
    else:
        y=1
    
    if (id_client is not None) & (parameter is not None):
        
        dv=df_client[parameter][df_client['TARGET'] == y].value_counts()
        dv=pd.DataFrame(dv)
        dv=dv.reset_index().reset_index()
        
        val = df_client[parameter][df_client.SK_ID_CURR == id_client].to_numpy()

        val=val[0]
      
        dv['level_0'][dv['index'] == val] = -1
        dv = dv.sort_values(by='level_0')
        dv['label'] = dv[parameter]
        #print(dv)

        colors = ['lightgrey','tan','lightyellow', 'navajowhite','salmon','coral']

        fig = go.Figure(data=[go.Pie(labels=dv['index'],pull=[0.1, 0, 0, 0,0,0],
                             values=dv['label'], name="World GDP 1980")])
        fig.update_traces(marker=dict(colors=colors))
        #fig.update_yaxes(automargin=True) 
        fig.update_layout(margin=dict(l=50, r=0, t=0, b=50))
        fig.update_layout(height=400,
                          title_font_size=14, title_font_color="black",
                          legend=dict(
                                        #orientation="h",
                                        yanchor="bottom",
                                        y=1.02,
                                        xanchor="right",
                                        x=1
                        ))
        fig.update_layout(legend_title_text="Classe du CLIENT: <b>{}<b>".format(val))

        return fig
    else:
        fig = {}
        return {
                "layout": {
                "xaxis": {
                "visible": False
                            },
                "yaxis": {
                "visible": False
                            },
                "annotations": [
                        {
                            "text": "No matching data found",
                            "xref": "paper",
                            "yref": "paper",
                            "showarrow": False,
                            "font": {
                            "size": 18
                }
            }
        ]
    }
}



#bar plot num client
@app.callback(Output('bar_plot', 'figure'), 
              Input('dropdown_client', 'value'),
              Input('dropdown_num','value'),
              Input('dropdown_cat','value'),
              Input('check_num','value')
             )
def bar_plot(id_client, parameter,cat,check):
     
    
    titre = ''   
    pformat='%{text:,.2f}'
    
    if (id_client is not None) & (parameter is not None):
        val =''
        x = ['<b>CLIENT<b>', '<b>EN DEFAUT<b>', 
             '<b>SOLVABLES<b>']
        y = [df_client[parameter][df_client.SK_ID_CURR == id_client].values[0],
                df_client[parameter][df_client.TARGET == 1].mean(),df_client[parameter][df_client.TARGET == 0].mean()]
        if (check == ['yes']) &  (cat is not None):  
            
            val = df_client[cat][df_client.SK_ID_CURR == id_client].to_numpy()
            val=val[0]
            titre = "<br> avec " + cat + ":" +val
            
            y = [df_client[parameter][df_client.SK_ID_CURR == id_client].values[0],
                 df_client[parameter][(df_client.TARGET == 1) & (df_client[cat]==val) ].mean(),
                 df_client[parameter][(df_client.TARGET == 0)  & (df_client[cat]==val)].mean()]
        else:
                y = [df_client[parameter][df_client.SK_ID_CURR == id_client].values[0],
                df_client[parameter][df_client.TARGET == 1].mean(),df_client[parameter][df_client.TARGET == 0].mean()]


        if df_client[parameter].dtype == np.int64:
            pformat == '%{text:,.0f}' 

        
        fig = go.Figure()
        fig.add_trace(go.Bar(
        x=x,
        y=y,
        text=y,
        textposition='outside',
        texttemplate=pformat,
        width=[0.6, 0.6,0.6],
        marker=dict(
        #color='rgba(50, 171, 96, 0.6)',
            color=["grey",rouge,bleue],
            opacity = [0.8,0.8,0.5],
            line=dict(
                color='rgba(50, 171, 96, 1.0)',
                width=0)
        )))
        #fig.update_xaxes(title_text=parameter)
        fig.update_yaxes(title_text='valeur')
        fig.layout.plot_bgcolor = "white"
        fig.update_xaxes(showline=False, linewidth=2, linecolor='black',gridcolor='white',)
        fig.update_yaxes(showline=False, linewidth=2, linecolor='black',gridcolor='lightgrey', zeroline=True, zerolinewidth=2, zerolinecolor='black')
        #fig.update_yaxes(automargin=True)    
        fig.update_layout(margin=dict(l=100, r=10, t=50, b=50),height=400)
        fig.update_layout(title_text="Valeur du client comparée aux valeurs moyennes "+titre,
                          title_font_size=14, title_font_color="black")
        return fig
    #height = 300,
    else:
        return {
    "layout": {
        "xaxis": {
            "visible": False
        },
        "yaxis": {
            "visible": False
        },
        "annotations": [
            {
                "text": "No matching data found",
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {
                    "size": 18
                }
            }
        ]
    }
}


                                                  
#indicateur de probalilité
@app.callback(Output('indicateur', 'figure'), 
              Input('dropdown_client', 'value'))
def indicateur(selected_value):
    
    if selected_value is not None:
            proba = lgbm.predict(df_client[selected_features][df_client.SK_ID_CURR == selected_value])[0]
    else:
            proba = 0
    
    fig = go.Figure(go.Indicator(
    mode = "number+gauge",
    gauge= {'shape': "bullet",
             'axis': {'range': [None, 100]},
    
             'bar': {'color': rouge,'thickness': 0.60},
             'steps' : [{'range': [49.4,100], 'color': color}],
             'threshold': {
                'line': {'color': rouge, 'width': 2},
                'thickness': 0.8, 'value':73},
                         },
    delta = {'reference': 150},
    value = proba*100,
    domain = {'x': [0.1, 1], 'y': [0, 1]},
    title = {'text': "%"}))
    fig.update_layout(height = 70,margin=dict(l=20, r=0, t=0, b=20))
    return fig

#call back affichage des fp et fn
@app.callback(Output('display_loan', 'children'), 
              Input('dropdown_client', 'value'))
def display_loan(selected_value):
         
        if selected_value is not None:
    
            loan_val = df_client['AMT_CREDIT'][df_client.SK_ID_CURR == selected_value].values.astype(int)[0]
            loan_price = df_client['AMT_GOODS_PRICE'][df_client.SK_ID_CURR == selected_value].values.astype(int)[0]
            loan_annuity = df_client['AMT_ANNUITY'][df_client.SK_ID_CURR == selected_value].values.astype(int)[0]
            loan_lenght = df_client['LENGTH_CREDIT'][df_client.SK_ID_CURR == selected_value].values.astype(int)[0]
            loan_cnt = df_client['CNT_ACTIVE_LOAN'][df_client.SK_ID_CURR == selected_value].values.astype(int)[0]
    
           

            return ('''
> ###### Montant emprunté   : __{:,.0f}__  
> - Mensualité        : {:,.0f}      
> - Montant à financer: {:,.0f}   
> - Durée             : {:,.1f}  
> - Nombre de crédit  : {:,.0f}  
      
'''.format(loan_val,loan_annuity,loan_price,loan_lenght,loan_cnt))
        else:
            loan_val =0
            
            return ('''    
   
  ''')    



#call back affichage des fp et fn
@app.callback(Output('disp_prob', 'children'), 
              Input('dropdown_client', 'value'))
def display_proba(selected_value):
    
        loss_given_def = 0.7 #arbitraire à fixer avec le metier
        gain_no_def = 0.2 #arbitraire à fixer avec le metier
        threshold = 0.735 #proba de perte minimale calculee sur le jeu de test        
        if selected_value is not None:
    
            loan_val = df_client['AMT_CREDIT'][df_client.SK_ID_CURR == selected_value].values.astype(int)[0]
            
            proba = lgbm.predict(df_client[selected_features][df_client.SK_ID_CURR == selected_value])[0]
            y = df_client.TARGET
    
            y_pred_prob = lgbm.predict(df_client[selected_features])
    
            opp = 0
            real = 0
            total = 0
        
            
            #calcul pour la probabilité de defaut du client
            prediction = y_pred_prob > proba

            cm = confusion_matrix(y, prediction)
            
            pcm = cm/np.sum(cm)
            
            pfp = pcm[1,0]
            pfn = pcm[0,1]

            return ('''     
> ###### Pour un seuil de défaut de __{:,.1f}%__  
> - Mauvaise prédiction de solvabilité : {:,.0f}%  
> - Mauvaise prédiction de défaut: {:,.0f}%  
'''.format(proba*100,pfp*100,pfn*100))
        else:
            loan_val =0
            pfp = 0
            pfn = 0
            proba = 0
            
            return (''' 
   
  ''')

# Tableau de données client
@app.callback(Output('tableau', 'figure'), 
              Input('dropdown_client', 'value'))
def update_table(id_client):
    pd.options.display.float_format = '{:.2f}%'.format
    
    if id_client is not None:

        fig = go.Figure(data=[go.Table(
        header=dict(values= [
                         "<b>GENRE</b>",
                          "<b>SITUATION</b>",
                         "<b>AGE</b>",
                         "<b>ENFANTS</b>", 
                         "<b>REVENU TOTAL</b>", 
                         "<b>TYPE DE REVENU</b>",
                         "<b>ANCIENNETE</b>",
                            ],
                    fill_color='black', line_color='white',
                    align='center',font=dict(color='white', size=10),height=40),
        cells=dict(values = [
                       
                       df_client[df_client.SK_ID_CURR == id_client].CODE_GENDER,
                       df_client[df_client.SK_ID_CURR == id_client].NAME_FAMILY_STATUS,
                       df_client[df_client.SK_ID_CURR == id_client].YEARS_BIRTH.values.astype(int),
                       df_client[df_client.SK_ID_CURR == id_client].CNT_CHILDREN, 
                       df_client[df_client.SK_ID_CURR == id_client].AMT_INCOME_TOTAL, 
                       df_client[df_client.SK_ID_CURR == id_client].NAME_INCOME_TYPE, 
                       df_client[df_client.SK_ID_CURR == id_client].YEARS_EMPLOYED.apply('{:.1f}'.format),
                            ],
                   fill_color=color,line_color='white',
                   font=dict(color='black', size=12),height=40,align=['center'],))
        ])
        fig.update_layout(height = 100,margin=dict(l=0, r=20, t=0, b=0))

        return fig
    else:
        fig = go.Figure(data=[go.Table(
        header=dict(values= [
                         "<b>GENRE</b>",
                          "<b>SITUATION</b>",
                         "<b>AGE</b>",
                         "<b>ENFANTS</b>", 
                         "<b>REVENU TOTAL</b>", 
                         "<b>TYPE DE REVENU</b>",
                         "<b>ANCIENNETE</b>"],
                    fill_color='black', line_color='white',
                    align='center',font=dict(color='white', size=10),height=40),
        cells=dict(values=['','','','','','',''],
                   fill_color=color,line_color='white',
                   font=dict(color='white', size=12),height=40,align=['center'],))
        ])
        fig.update_layout(height = 100,margin=dict(l=0, r=0, t=0, b=0))


        return fig
        
#app.run_server(mode='external',debug=True, use_reloader=False)        
#app.run_server(mode='inline', port=8069)

if __name__ == '__main__':
    app.run_server(debug=True)