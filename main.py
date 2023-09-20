import pandas as pd
import sys
import numpy as np
import io
import streamlit as st
import funcs
from PIL import Image
SEED = 42
logo = Image.open('img/logo.png')
# Define as cores
cor_letra = '#ff7131'
cor_topicos = '#fe3d67'
if 'dados_carregados' not in st.session_state:
    st.session_state['dados_carregados'] = []
    st.session_state['exportar'] = []
    st.session_state['dados_transformados'] = []
    st.session_state['classificador_selecionado'] = []
    st.session_state['classificadores'] = []
def transformar_dados(dados):
    dados = dados.rename({'customerID':'id_cliente'},axis=1)
    # Traduzindo e descompactando colunas
    for k, v in dados.iterrows():
        # Dados Cliente
        dados.loc[k, 'cliente_genero'] = v.customer['gender']
        dados.loc[k, 'flagIdoso'] = v.customer['SeniorCitizen']
        dados.loc[k, 'flagParceiro'] = v.customer['Partner']
        dados.loc[k, 'flagDependentes'] = v.customer['Dependents']
        dados.loc[k, 'duracaoContrato_meses'] = v.customer['tenure']
        # Dados Telefone
        dados.loc[k, 'flagAssinaturaTel'] = v.phone['PhoneService']
        dados.loc[k, 'flagAssinaturaTelMultiplasLinhas'] = v.phone['MultipleLines']
        # Dados Internet
        dados.loc[k, 'internetTipo'] = v.internet['InternetService']
        dados.loc[k, 'flagAssinaturaSeguranca'] = v.internet['OnlineSecurity']
        dados.loc[k, 'flagAssinaturaBackup'] = v.internet['OnlineBackup']
        dados.loc[k, 'flagAssinaturaProtecaoMovel'] = v.internet['DeviceProtection']
        dados.loc[k, 'flagAssinaturaSuporteTecnico'] = v.internet['TechSupport']
        dados.loc[k, 'flagAssinaturaTV'] = v.internet['StreamingTV']
        dados.loc[k, 'flagAssinaturaFilmes'] = v.internet['StreamingMovies']
        # Dados Contrato
        dados.loc[k, 'contratoTipo'] = v.account['Contract']
        dados.loc[k, 'flagFaturaOnline'] = v.account['PaperlessBilling']
        dados.loc[k, 'pagamentoTipo'] = v.account['PaymentMethod']
        dados.loc[k, 'custoMensal'] = v.account['Charges']['Monthly']
        dados.loc[k, 'custoMontante'] = v.account['Charges']['Total']

    # Apagando as variáveis traduzidas e descompactadas
    colunas_dic = ['customer', 'phone', 'internet', 'account']
    for i in colunas_dic:
        dados = dados.drop(columns=i)

    # Transformando True e False em binários
    # Transformando dados em variáveis binárias, exceto o transformado anteriormente.
    dados['Churn'] = dados['Churn'].map({'No': 0, 'Yes': 1})
    dados['flagParceiro'] = dados['flagParceiro'].map({'No': 0, 'Yes': 1})
    dados['flagDependentes'] = dados['flagDependentes'].map({'No': 0, 'Yes': 1})
    dados['flagAssinaturaTel'] = dados['flagAssinaturaTel'].map({'No': 0, 'Yes': 1})
    dados['flagAssinaturaTelMultiplasLinhas'] = dados['flagAssinaturaTelMultiplasLinhas'].map({'No': 0, 'Yes': 1})
    dados['flagAssinaturaSeguranca'] = dados['flagAssinaturaSeguranca'].map({'No': 0, 'Yes': 1})
    dados['flagAssinaturaBackup'] = dados['flagAssinaturaBackup'].map({'No': 0, 'Yes': 1})
    dados['flagAssinaturaProtecaoMovel'] = dados['flagAssinaturaProtecaoMovel'].map({'No': 0, 'Yes': 1})
    dados['flagAssinaturaSuporteTecnico'] = dados['flagAssinaturaSuporteTecnico'].map({'No': 0, 'Yes': 1})
    dados['flagAssinaturaTV'] = dados['flagAssinaturaTV'].map({'No': 0, 'Yes': 1})
    dados['flagAssinaturaFilmes'] = dados['flagAssinaturaFilmes'].map({'No': 0, 'Yes': 1})
    dados['flagFaturaOnline'] = dados['flagFaturaOnline'].map({'No': 0, 'Yes': 1})
    dados['flagIdoso'] = dados['flagIdoso'].map({0.0: 0, 1.0: 1})
    dados['custoMensal'] = dados['custoMensal'].astype(int)
    dados['custoMontante'] = dados['custoMontante'].str.strip().replace('', np.NaN)
    dados['custoMontante'] = dados['custoMontante'].astype(float).fillna(0)

    # Transformando dados em variáveis binárias, exceto o transformado anteriormente.
    dados_dummies = pd.get_dummies(dados.drop(
        ['Churn', 'flagParceiro', 'flagDependentes', 'flagAssinaturaTel', 'flagAssinaturaTelMultiplasLinhas',
         'flagAssinaturaSeguranca',
         'flagAssinaturaBackup', 'flagAssinaturaProtecaoMovel', 'flagAssinaturaSuporteTecnico', 'flagAssinaturaTV',
         'flagAssinaturaFilmes',
         'flagFaturaOnline', 'flagIdoso', 'id_cliente'], axis=1))

    dados_full = pd.concat([dados[['Churn', 'flagParceiro', 'flagDependentes', 'flagAssinaturaTel',
                                   'flagAssinaturaTelMultiplasLinhas', 'flagAssinaturaSeguranca',
                                   'flagAssinaturaBackup', 'flagAssinaturaProtecaoMovel',
                                   'flagAssinaturaSuporteTecnico', 'flagAssinaturaTV', 'flagAssinaturaFilmes',
                                   'flagFaturaOnline', 'flagIdoso', 'id_cliente']], dados_dummies], axis=1)

    ## Preenchendo vazios com mediana e transformando clienteId em index
    dados_full = dados_full.set_index('id_cliente',drop=True)
    dados_full = dados_full.fillna(dados_full.median())

    # Selecionando as variáveis com maior correlação com a Churn
    correlacoes = dados_full.corr()['Churn']
    variaveis_validas = correlacoes[correlacoes.abs() > 0.15].index.tolist()
    dados_correlatos = dados_full[variaveis_validas]
    return dados_correlatos
def selecionar_classificador(dados_transformados):
    X = dados_transformados.drop('Churn', axis=1)
    y = dados_transformados['Churn']
    classificadores = funcs.classificadores(X,y)
    return classificadores
def carregando(arquivo, link):
    if arquivo is not None:
        if arquivo.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            dados = pd.read_excel(arquivo)
        elif arquivo.type == 'application/json':
            dados = pd.read_json(arquivo)
        else:
            st.error("Erro ao importar, arquivo não suportado!")
            st.cache_data.clear()
            st.experimental_rerun()
    elif link:
        try:
            dados = pd.read_excel(link) if link.endswith('.xlsx') else pd.read_json(link)
        except:
            st.error('Erro ao importar dados do link. Certifique-se de que o link seja válido e aponte para um arquivo XLSX ou JSON.')

            st.cache_data.clear()
            st.experimental_rerun()
            return None
    else:
        st.error('Por favor, carregue um arquivo ou informe um link.')
        return None
    dados_transformados = transformar_dados(dados)
    return dados_transformados

st.image(logo,width=300)
st.markdown(f"""
    <style>
        .titulo {{
            color: {cor_topicos};
        }}
        p {{
            color: {cor_letra};
        }}
    </style>
    <h1 class="titulo">Novexus AntiChurn</h1>
    <p>Bem vindo ao sistema de prevenção de Churn da Novexus, importe ou cole a url dos dados em Json, no formato padrão para que o sistema analise os dados e classifique os clientes para você iniciar as tratativas de prevenção</p>""",unsafe_allow_html=True)
st.divider()
def main():
    campo1 = st.empty()
    campo2 = st.empty()
    campo3 = st.empty()
    campo4 = st.empty()
    with campo1.container():
        st.markdown(f"""
                <style>
                    .titulo {{
                        color: {cor_topicos};
                    }}
                    p {{
                        color: {cor_letra};
                    }}
                </style>
                <h3 class="titulo">Carregue os dados dos clientes em Json ou XLSX</h3>""", unsafe_allow_html=True)
        arquivo = st.file_uploader('Importar arquivo XLSX ou JSON', type=['xlsx', 'json'])
        link = st.text_input('Importar dados a partir de um link')
        if st.button('Carregar Dados'):
            st.session_state['dados_carregados'] = True
    #Entrando nas sessões
    if st.session_state.dados_carregados:
        with campo2.container():
            st.warning('Aguarde enquanto processo seus dados.')
            dados_transformados = carregando(arquivo, link)
            st.session_state['dados_transformados'] = dados_transformados
            if dados_transformados is not None:
                st.success(f"Foram carregados {len(dados_transformados)} clientes")
                st.write('**Dados Convertidos:**')
                st.write(dados_transformados.head())
            b_prever = st.button('Churn - Classificar dados')
            if b_prever:
                campo1.warning("Gerando tabela... Aguarde")
                campo2.empty()
                st.session_state['exportar'] = True
    if st.session_state.exportar == True:
        campo1.empty()
        campo2.empty()
        st.session_state.dados_carregados = False
        classificadores = st.session_state['classificadores']
        if len(classificadores) == 0:
            classificadores = selecionar_classificador(st.session_state['dados_transformados'])
        else:
            pass
        with campo3.container():
            st.session_state['classificadores'] = classificadores
            classificador_selecionado = st.radio("Escolha o classificador:",
                         [x for x in classificadores.keys()],
                         captions = [x for x in [v['Avaliação'] for k, v in classificadores.items()]],horizontal=True)
            with st.expander("Veja o que significa cada métrica."):
                st.markdown("<h4>Acurácia:</h4> Indica a performance geral do modelo, ou seja, a porcentagem de classificações corretas em relação ao total de classificações. No contexto de evitar o Churn, a acurácia mede a capacidade do modelo de prever corretamente se um cliente irá ou não cancelar o serviço. Uma acurácia alta significa que o modelo está fazendo boas previsões no geral.<br>"
                     "<h4>Precisão:</h4> Mede a proporção de verdadeiros positivos (clientes que o modelo previu corretamente que iriam cancelar) em relação ao total de clientes que o modelo classificou como positivos (clientes que o modelo previu que iriam cancelar, independentemente de estarem corretos ou não). No contexto de evitar o Churn, uma precisão alta significa que o modelo está fazendo poucos erros do tipo I, ou seja, prevendo corretamente os clientes que realmente irão cancelar.<br>"
                     "<h4>Recall:</h4> Mede a proporção de verdadeiros positivos em relação ao total de clientes que realmente cancelaram. No contexto de evitar o Churn, um recall alto significa que o modelo está capturando a maioria dos clientes que realmente irão cancelar, minimizando os erros do tipo II, ou seja, prevendo corretamente os clientes que não irão cancelar.<br>"
                     "<h4>F1-Score:</h4> É uma métrica que combina precisão e recall de maneira equilibrada, fornecendo uma medida única de desempenho do modelo. No contexto de evitar o Churn, um F1-Score alto indica que o modelo está fazendo um bom trabalho em prever tanto os clientes que irão cancelar quanto os que não irão, sem sacrificar uma métrica pela outra<br><br>",unsafe_allow_html=True)
            classificador_on = st.button(f"Prever com {classificador_selecionado}")
            if classificador_on:
                st.session_state['classificador_selecionado'] = classificador_selecionado
    #Processo de exportação dos dados.
    if st.session_state['classificador_selecionado'] == 'KNN':
        campo2.empty()
        st.session_state['dados_carregados'] = False
        dados_transformados = st.session_state['dados_transformados']
        classificador = st.session_state['classificador_selecionado']
        with campo4.container():
            dados_transformados['Probabilidade_de_Churn'] = classificadores[classificador]['Prob']
            dados_transformados['Probabilidade_de_Churn'] = dados_transformados['Probabilidade_de_Churn'].apply(lambda x: f"{round(x*100,2)}%")
            dados_transformados['Prob'] = classificadores[classificador]['Prob']
            st.write('**Ranking de Potenciais (com mais de 60%) Clientes que Desejam Sair da Empresa:**')
            st.warning(
                f"Tendo em vista que já  perdemos {len(dados_transformados[dados_transformados['Churn'] == 1])} clientes")
            st.write(dados_transformados.loc[
                                 (dados_transformados['Churn'] == 0) & (dados_transformados['Prob'] >= 0.6), [
                                     'Probabilidade_de_Churn']])
            dados_transformados[(dados_transformados['Churn'] == 0) & (dados_transformados['Prob'] >= 0.6)].to_excel('Novexus_Possivel_Churn_KNN.xlsx',index=True)
            try:
                with open('Novexus_Possivel_Churn_KNN.xlsx', 'rb') as f:
                    bytes_data = f.read()
                    st.download_button(
                        label='Baixar dados',
                        data=bytes_data,
                        file_name='Novexus_Possivel_Churn_KNN.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )
                st.cache_data.clear()
                st.experimental_rerun()
            except:
                st.experimental_rerun()
    elif st.session_state['classificador_selecionado'] == 'BNB':
        campo2.empty()
        st.session_state['dados_carregados'] = False
        dados_transformados = st.session_state['dados_transformados']
        classificador = st.session_state['classificador_selecionado']
        with campo4.container():
            dados_transformados['Probabilidade_de_Churn'] = classificadores[classificador]['Prob']
            dados_transformados['Probabilidade_de_Churn'] = dados_transformados['Probabilidade_de_Churn'].apply(lambda x: f"{round(x * 100, 2)}%")
            dados_transformados['Prob'] = classificadores[classificador]['Prob']
            st.write('**Ranking de Potenciais (com mais de 60%) Clientes que Desejam Sair da Empresa:**')
            st.warning(
                f"Tendo em vista que já  perdemos {len(dados_transformados[dados_transformados['Churn'] == 1])} clientes")
            st.write(dados_transformados.loc[
                         (dados_transformados['Churn'] == 0) & (dados_transformados['Prob'] >= 0.6), [
                             'Probabilidade_de_Churn']])
            dados_transformados[(dados_transformados['Churn'] == 0) & (dados_transformados['Prob'] >= 0.6)].to_excel('Novexus_Possivel_Churn_BNB.xlsx', index=True)
            try:
                with open('Novexus_Possivel_Churn_BNB.xlsx', 'rb') as f:
                    bytes_data = f.read()
                    st.download_button(
                        label='Baixar dados',
                        data=bytes_data,
                        file_name='Novexus_Possivel_Churn_BNB.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )
                st.cache_data.clear()
                st.experimental_rerun()
            except:
                st.experimental_rerun()
    elif st.session_state['classificador_selecionado'] == 'DTC':
        campo2.empty()
        st.session_state['dados_carregados'] = False
        dados_transformados = st.session_state['dados_transformados']
        classificador = st.session_state['classificador_selecionado']
        with campo4.container():
            dados_transformados['Probabilidade_de_Churn'] = classificadores[classificador]['Prob']
            dados_transformados['Probabilidade_de_Churn'] = dados_transformados['Probabilidade_de_Churn'].apply(lambda x: f"{round(x * 100, 2)}%")
            dados_transformados['Prob'] = classificadores[classificador]['Prob']
            st.write('**Ranking de Potenciais (com mais de 60%) Clientes que Desejam Sair da Empresa:**')
            st.warning(
                f"Tendo em vista que já  perdemos {len(dados_transformados[dados_transformados['Churn'] == 1])} clientes")
            st.write(dados_transformados.loc[
                         (dados_transformados['Churn'] == 0) & (dados_transformados['Prob'] >= 0.6), [
                             'Probabilidade_de_Churn']])
            dados_transformados[(dados_transformados['Churn'] == 0) & (dados_transformados['Prob'] >= 0.6)].to_excel('Novexus_Possivel_Churn_DTC.xlsx', index=True)
            try:
                with open('Novexus_Possivel_Churn_DTC.xlsx', 'rb') as f:
                    bytes_data = f.read()
                    st.download_button(
                        label='Baixar dados',
                        data=bytes_data,
                        file_name='Novexus_Possivel_Churn_DTC.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )
                st.cache_data.clear()
                st.experimental_rerun()
            except:
                st.experimental_rerun()


if __name__ == '__main__':
    main()
