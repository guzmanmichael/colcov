import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

import pandas as pd
import json
import requests
import numpy as np
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go

app = dash.Dash(__name__, title='COVStats',  external_stylesheets=[dbc.themes.MINTY])

#Consulta API:
url = 'https://services.arcgis.com/BQTQBNBsmxjF8vus/ArcGIS/rest/services/Colombia_COVID19V/FeatureServer/6/query?where=1%3D1&outFields=*&outSR=4326&f=json'
data = requests.get(url)
data = data.json()
data = data['features']

#Limpieza y creación de Dataframe:

n_days = len(data)
attributes = data[0]['attributes']

NUEVOS_CASOS = []
NUEVOS_MUERTOS = []
NUEVOS_RECUPERADOS = []
DIAS = []
TOTAL_CASOS = []
TOTAL_MUERTES = []
TOTAL_RECUPERADOS = []
for i in range(n_days):
  NUEVOS_CASOS.append(data[i]['attributes']['NUEVOS_CASOS'])
  NUEVOS_MUERTOS.append(data[i]['attributes']['NUEVOS_MUERTOS'])
  NUEVOS_RECUPERADOS.append(data[i]['attributes']['NUEVO_RECUPERADOS'])
  DIAS.append(i+1)

  TOTAL_CASOS.append(data[i]['attributes']['TOTAL_CASOS'])
  TOTAL_MUERTES.append(data[i]['attributes']['TOTAL_MUERTES'])
  TOTAL_RECUPERADOS.append(data[i]['attributes']['TOTAL_RECUPERADOS'])

  df = {
    'Casos': NUEVOS_CASOS,
    'Muertos': NUEVOS_MUERTOS,
    'Recuperados': NUEVOS_RECUPERADOS,
    'Día': DIAS,
    'TL': np.array(NUEVOS_MUERTOS)/np.array(NUEVOS_CASOS)
}

df_a = {
    'Casos': TOTAL_CASOS,
    'Muertos': TOTAL_MUERTES,
    'Recuperados': TOTAL_RECUPERADOS,
    'Día': DIAS
}

df_g = {
    'Casos': df_a['Casos'][-1],
    'Muertos': df_a['Muertos'][-1],
    'Recuperados': df_a['Recuperados'][-1],
}

stats = {
    'Tasa de letalidad': df_g['Muertos']/df_g['Casos'],
    'CPM': (df_g['Casos']/49397318)*1000000

}

df = pd.DataFrame(data=df)
df_a = pd.DataFrame(data=df_a)

#Gráficos:
x_bar = ['Confirmados', 'Recuperados', 'Muertos']
y_bar = [df_g['Casos'], df_g['Recuperados'], df_g['Muertos']]

fig = go.Figure(data=[go.Bar(x=x_bar, y=y_bar, marker_color=['blue', 'green', 'red'])], layout=go.Layout( title=go.layout.Title(text="Balance general - COVID-19, Colombia")))
##########################################################################################
fig1 = go.Figure(layout=go.Layout(title=go.layout.Title(text="Acumulados por día - COVID-19, Colombia")))
fig1.add_trace(go.Scatter(x=df_a['Día'], y=df_a['Casos'], fill='tozeroy', name='Confirmados'))
fig1.add_trace(go.Scatter(x=df_a['Día'], y=df_a['Recuperados'], fill='tozeroy', name='Recuperados')) 
fig1.add_trace(go.Scatter(x=df_a['Día'], y=df_a['Muertos'], fill='tozeroy', name='Muertos')) 
##########################################################################################

fig2 = px.bar(df, x='Día', y='Muertos')

fig2 = go.Figure(data=[
    go.Bar(name='Confirmados', x=df['Día'], y=df['Casos']),
    go.Bar(name='Recuperados', x=df['Día'], y=df['Recuperados']),
    go.Bar(name='Muertos', x=df['Día'], y=df['Muertos'])
  ],     
  layout=go.Layout( title=go.layout.Title(text="Variación por día - COVID-19, Colombia")))
##########################################################################################
fig3 = go.Figure(layout=go.Layout(title=go.layout.Title(text="Letalidad por día - COVID-19, Colombia")))
fig3.add_trace(go.Bar(name='Letalidad por día reportado' ,x=df['Día'], y=df['TL']))
##########################################################################################
zc = np.polyfit(df['Día'], df['Casos'], 9)
fc = np.poly1d(zc)

fig4 = go.Figure(layout=go.Layout(title=go.layout.Title(text="Tendencia en casos confirmados - COVID-19, Colombia")))
fig4.add_trace(go.Scatter(name='Confirmados', x=df['Día'], y=df['Casos']))
fig4.add_trace(go.Scatter(name='Tendencia' ,x=df['Día'], y=fc(df['Día'])))
##########################################################################################
zm = np.polyfit(df['Día'], df['Muertos'], 9)
fm = np.poly1d(zm)

fig5 = go.Figure(layout=go.Layout(title=go.layout.Title(text="Tendencia en muertes - COVID-19, Colombia")))
fig5.add_trace(go.Scatter(name='Confirmados', x=df['Día'], y=df['Muertos']))
fig5.add_trace(go.Scatter(name='Tendencia' ,x=df['Día'], y=fm(df['Día'])))
##########################################################################################
zr = np.polyfit(df['Día'], df['Recuperados'], 11)
fr = np.poly1d(zr)

fig6 = go.Figure(layout=go.Layout(title=go.layout.Title(text="Tendencia en recuperados - COVID-19, Colombia")))
fig6.add_trace(go.Scatter(name='Confirmados', x=df['Día'], y=df['Recuperados']))
fig6.add_trace(go.Scatter(name='Tendencia' ,x=df['Día'], y=fr(df['Día'])))
##########################################################################################
#Creación de la app


app.layout = dbc.Container(
    [

    dbc.Jumbotron([
        html.H1("Datos COVID-19. Colombia", className="display-3"),
        html.P(
            "Creado por Michael Guzmán de las Salas",
            className="lead",
        )
    ]),

    dbc.Alert("Los datos están sujetos a las actualizaciones del gobierno nacional.", color="success"),

    dbc.Button(
    ["Casos por millón de habitantes: ", dbc.Badge(round(stats['CPM']), color="info", className="m-2")],
    color="info", className="m-2"
    ),

    dbc.Button(
    ["Tasa de letalidad (%): ", dbc.Badge(f"{round(stats['Tasa de letalidad']*100, 2)}%", color="info", className="m-2")],
    color="info", className="m-2"
    ),

    dbc.Button(
    ["Casos nuevos: ", dbc.Badge(NUEVOS_CASOS[-1], color="warning", className="m-2")],
    color="warning", className="m-2"
    ),

    dbc.Button(
    ["Muertes hoy: ", dbc.Badge(NUEVOS_MUERTOS[-1], color="danger", className="m-2")],
    color="danger", className="m-2"
    ),

    dbc.Button(
    ["Recuperados hoy: ", dbc.Badge(NUEVOS_RECUPERADOS[-1], color="success", className="m-2")],
    color="success", className="m-2"
    ),

    dcc.Graph(figure=fig),
    dcc.Graph(figure=fig1),
    dcc.Graph(figure=fig2),
    dcc.Graph(figure=fig3),
    dcc.Graph(figure=fig4),
    dcc.Graph(figure=fig5),
    dcc.Graph(figure=fig6)
    ]
)

if __name__ == '__main__':
    app.run_server(debug=True)