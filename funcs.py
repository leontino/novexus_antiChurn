def classificadores(x,y):
    # Importações
    import pickle
    import os
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    # Carregando modelos
    knn_file = 'classificadores/knn.pkl'
    bnb_file = 'classificadores/bnb.pkl'
    dtc_file = 'classificadores/dtc.pkl'
    knn = None
    bnb = None
    dtc = None
    if os.path.isfile(knn_file):
        with open(knn_file, 'rb') as f:
            knn = pickle.load(f)
    if os.path.isfile(bnb_file):
        with open(bnb_file, 'rb') as f:
            bnb = pickle.load(f)
    if os.path.isfile(dtc_file):
        with open(dtc_file, 'rb') as f:
            dtc = pickle.load(f)
    # Dados
    norm = StandardScaler()
    X_normalizado = norm.fit_transform(x)
    X_treino, X_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=42)

    #KNN
    param_grid = {'n_neighbors': range(1, 30)}
    if knn is None:
        knn = KNeighborsClassifier()
        grid_search = GridSearchCV(knn, param_grid, cv=5)
        grid_search.fit(X_normalizado, y)
        melhor_vizinhos = grid_search.best_params_['n_neighbors']
        knn = KNeighborsClassifier(metric='euclidean', n_neighbors=melhor_vizinhos)
        knn.fit(X_treino, y_treino)
    knn_y_pred = knn.predict(X_teste)
    knn_y_prob = knn.predict_proba(x)
    knn_accuracy = accuracy_score(y_teste, knn_y_pred)
    knn_precision = precision_score(y_teste, knn_y_pred, average='weighted')
    knn_recall = recall_score(y_teste, knn_y_pred, average='weighted')
    knn_f1 = f1_score(y_teste, knn_y_pred, average='weighted')
    avaliacao_knn = f"""Modelo KNeighborsClassifier:\n
    Acurácia: {round(knn_accuracy*100,2)}%\n
    Precisão: {round(knn_precision*100,2)}%\n
    Recall: {round(knn_recall*100,2)}%\n
    F1-Score: {round(knn_f1*100,2)}%"""
    #BerdolinniNB
    param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]}
    if bnb is None:
        bnb = BernoulliNB()
        grid_search = GridSearchCV(bnb, param_grid, cv=5)
        grid_search.fit(X_treino, y_treino)
        best_alpha = grid_search.best_params_['alpha']
        bnb = BernoulliNB(alpha=best_alpha)
        bnb.fit(X_treino, y_treino)
    bnb_y_pred = bnb.predict(X_teste)
    bnb_y_prob = bnb.predict_proba(x)
    bnb_accuracy = accuracy_score(y_teste, bnb_y_pred)
    bnb_precision = precision_score(y_teste, bnb_y_pred, average='weighted')
    bnb_recall = recall_score(y_teste, bnb_y_pred, average='weighted')
    bnb_f1 = f1_score(y_teste, bnb_y_pred, average='weighted')
    avaliacao_bnb = f"""Modelo BernoulliNB:\n
    Acurácia: {round(bnb_accuracy*100,2)}%\n
    Precisão: {round(bnb_precision*100,2)}%\n
    Recall: {round(bnb_recall*100,2)}%\n
    F1-Score: {round(bnb_f1*100,2)}%"""
    #DecisionTreeClassifier
    param_grid = {'max_depth': range(1, 10)}
    if dtc is None:
        dtc = DecisionTreeClassifier()
        grid_search = GridSearchCV(dtc, param_grid, cv=5)
        grid_search.fit(X_treino, y_treino)
        best_max_depth = grid_search.best_params_['max_depth']
        dtc = DecisionTreeClassifier(max_depth=best_max_depth)
        dtc.fit(X_treino, y_treino)
    dtc_y_pred = dtc.predict(X_teste)
    dtc_y_prob = dtc.predict_proba(x)
    dtc_accuracy = accuracy_score(y_teste, dtc_y_pred)
    dtc_precision = precision_score(y_teste, dtc_y_pred, average='weighted')
    dtc_recall = recall_score(y_teste, dtc_y_pred, average='weighted')
    dtc_f1 = f1_score(y_teste, dtc_y_pred, average='weighted')
    avaliacao_dtc = f"""Modelo DecisionTreeClassifier:\n
    Acurácia: {round(dtc_accuracy*100,2)}%\n
    Precisão: {round(dtc_precision*100,2)}%\n
    Recall: {round(dtc_recall*100,2)}%\n
    F1-Score: {round(dtc_f1*100,2)}%"""

    resultados = {
        'KNN': {"Avaliação":avaliacao_knn,"Prob":[x[1] for x in knn_y_prob]},
        'BNB': {"Avaliação":avaliacao_bnb,"Prob":[x[1] for x in bnb_y_prob]},
        'DTC': {"Avaliação":avaliacao_dtc,"Prob":[x[1] for x in dtc_y_prob]},
    }

    with open('classificadores/knn.pkl', 'wb') as f:
        pickle.dump(knn, f)
    with open('classificadores/bnb.pkl', 'wb') as f:
        pickle.dump(bnb, f)
    with open('classificadores/dtc.pkl', 'wb') as f:
        pickle.dump(dtc, f)

    return resultados
