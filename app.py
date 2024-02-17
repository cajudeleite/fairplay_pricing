import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
import tensorflow as tf
import streamlit as st


# Caminho para o arquivo CSV
caminho_csv = './brasileirao_serie_a.csv'

# Lista de nomes das colunas que você deseja selecionar
colunas_desejadas = ['data', 'rodada', 'estadio','publico', 'time_mandante', 'time_visitante','colocacao_mandante','colocacao_visitante', 'valor_equipe_titular_mandante', 'valor_equipe_titular_visitante']

# Carregar apenas as colunas desejadas do arquivo CSV para um DataFrame pandas
df = pd.read_csv(caminho_csv, usecols=colunas_desejadas)
df = df.loc[df['estadio'] == 'Allianz Parque']
df = df.loc[df['time_mandante'] == 'Palmeiras']

times_visitantes = df['time_visitante'].unique()

df['data'] = pd.to_datetime(df['data'])
df['publico_max'] = 43600
df = df.loc[df['publico'] != 0]
df = df.drop(columns=['estadio'])
df = df.drop(columns=['time_mandante'])

# Calcular os dias desde uma data de referência (por exemplo, 01/01/1970)
df['data'] = (df['data'] - pd.Timestamp('1970-01-01')).dt.days


for column in df.columns:
    if((df[column].isnull().sum()/len(df)) > 0.5):
        df = df.drop(columns=[column])

df = df.dropna()

df.reset_index(drop=True, inplace=True)

# Instantiate the OneHotEncoder
ohe_binary = OneHotEncoder(drop="if_binary")

# Fit encoder
ohe_binary.fit(df[['time_visitante']])

colunas_times_visitantes = ohe_binary.get_feature_names_out()

# Transformar os dados de codificação one-hot em um DataFrame
encoded_columns = pd.DataFrame(ohe_binary.transform(df[['time_visitante']]).toarray(), columns=ohe_binary.get_feature_names_out())

# Concatenar o DataFrame original com as colunas codificadas
df = pd.concat([df, encoded_columns], axis=1)


# Drop the column "time_visitante" which has been encoded
df.drop(columns = ["time_visitante"], inplace = True)

df.to_csv('./data.csv', index=False)



caminho_csv = './precos.csv'

# Lista de nomes das colunas que você deseja selecionar
colunas_desejadas = ['Arrecadação total', 'Total de ingressos vendidos']

preco_df = pd.read_csv(caminho_csv, usecols=colunas_desejadas)

# preco_df.rename(columns={'Data do jogo': 'data'}, inplace=True)
preco_df.rename(columns={'Arrecadação total': 'arrecadacao'}, inplace=True)
preco_df.rename(columns={'Total de ingressos vendidos': 'publico'}, inplace=True)


# preco_df['data'] = pd.to_datetime(preco_df['data'])
# preco_df['data'] = (preco_df['data'] - pd.Timestamp('1970-01-01')).dt.days

preco_df['publico'] = preco_df['publico'].str.replace('.', '')
preco_df['publico'] = preco_df['publico'].str.replace(',00', '')
preco_df['publico'] = preco_df['publico'].astype(int)

preco_df['arrecadacao'] = preco_df['arrecadacao'].str.replace('.', '')
preco_df['arrecadacao'] = preco_df['arrecadacao'].str.replace(',', '.')
preco_df['arrecadacao'] = preco_df['arrecadacao'].astype(float)


df = pd.merge(df, preco_df, on='publico', how='inner')

df['preco'] = df['arrecadacao'] / df['publico']

# df.drop(columns = ["arrecadacao"], inplace = True)

test = df.nlargest(3, 'data')
test = test.drop(columns=['publico'])


# Find the remaining rows
train = df.drop(test.index)

# Dividir os dados em features (X) e target (y)
X_train = train.drop(columns=['publico'])
y_train = train['publico']


# Inicializar o modelo de regressão linear
modelo_regressao_linear = LinearRegression()

# Treinar o modelo com os dados de treinamento
model_fit = modelo_regressao_linear.fit(X_train, y_train)

# # Fazer previsões usando o conjunto de teste
# y_pred = modelo_regressao_linear.predict(X_test)

# # Avaliar o desempenho do modelo usando a métrica Mean Squared Error (MSE)
# mse = mean_squared_error(y_test, y_pred)
# print('Mean Squared Error (MSE):', mse)

# r_squared = r2_score(y_test, y_pred)
# print('R-squared:', r_squared)



# # Create a new DataFrame with the same columns as df
# new_row = pd.DataFrame(columns=df.columns)

# # Fill the new row with zeros
# new_row.loc[0] = 0

# new_row.drop(columns = ["preco"], inplace = True)

# new_row['publico'] = 43600



st.markdown("""# Fairplay Pricing""")

preço = st.slider('Preço', 1, 100, 1)

test['preco'] = preço

prediction = model_fit.predict(test)

# time = st.empty().selectbox(
#     'Qual sera o time visitante?',
#     times_visitantes)
# coluna_time = [s for s in colunas_times_visitantes if time in s][0]
# new_row[coluna_time] = 1

# data = st.empty().date_input("Quando sera o jogo?")
# new_row['data'] = pd.to_datetime(data)
# new_row['data'] = (new_row['data'] - pd.Timestamp('1970-01-01')).dt.days

# rodada = st.slider('Qual sera a rodada?', 1, 32, 1)
# new_row['rodada'] = rodada

# coloc_pal = st.slider('Qual é a colocaçao do Palmeiras?', 1, 20, 1)
# new_row['colocacao_mandante'] = coloc_pal

# coloc_visitante = st.slider('Qual é a colocaçao do time visitante?', 1, 20, 1)
# new_row['colocacao_visitante'] = coloc_visitante

show = test
show['arrecadacao'] = prediction * preço
show['publico'] = prediction.round()

# Assuming 'df' is your DataFrame and 'specific_columns' is a list of column names
# you want to place at the start of the DataFrame
specific_columns = ['arrecadacao', 'preco', 'publico']

# Get the remaining columns
other_columns = [col for col in show.columns if col not in specific_columns]

# Reorder columns and concatenate them
new_order = specific_columns + other_columns
new_df = show[new_order]

new_df
