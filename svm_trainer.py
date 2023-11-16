from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time


class SVMTrainer:

    @staticmethod
    def svm_rbf(df, plotReport):
        # Divide los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(df["Features"].tolist(),
                                                            df["Label"].tolist(), test_size=0.2,
                                                            random_state=42)

        # Estandariza los datos
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Entrenar el modelo SVM lineal
        svm = SVC(kernel='rbf')

        # Entrenar el modelo
        svm.fit(X_train_scaled, y_train)

        # Predice las etiquetas en el conjunto de prueba
        y_pred = svm.predict(X_test_scaled)

        # Calcula la precisión del modelo
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='micro')

        if plotReport:
            print(f"Accuracy: {accuracy}")
            print(f"F1: {f1}")

            # Crear un DataFrame con la información del classification_report
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()

            # Configurar el estilo de la tabla con colores usando seaborn
            sns.set(font_scale=1)
            plt.figure(figsize=(10, 5))
            sns.heatmap(report_df.iloc[:-1, :-1], annot=True, fmt=".2f", cmap="Reds", cbar=False)
            plt.title("Classification Report")

            # Mostrar la figura
            plt.show()

    @staticmethod
    def svm_gridSearch(df, initial_time):
        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(df["Features"].tolist(),
                                                            df["Label"].tolist(), test_size=0.2,
                                                            random_state=42)
        # Estandarizar los datos
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Crear un modelo SVM lineal
        modelo_svm = SVC()

        # Definir los parámetros a buscar en la búsqueda exhaustiva

        parametros_grid_linear = {
            'kernel': ['linear'],
            'C': [0.1, 1, 10, 100],
        }

        parametros_grid_rbf = {
            'kernel': ['rbf'],
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 1, 0.1, 0.01, 0.001],
        }

        parametros_grid_poly = {
            'kernel': ['poly'],
            'C': [0.1, 1, 10, 100],
            'degree': [2, 3, 4],
            'coef0': [0.0, 1.0, 2.0],
        }

        parametros_grid_sigmoid = {
            'kernel': ['sigmoid'],
            'C': [0.1, 1, 10, 100],
            'coef0': [0.0, 1.0, 2.0],
            'gamma': ['scale', 'auto', 1, 0.1, 0.01, 0.001],
        }

        parametros_grid = [parametros_grid_linear, parametros_grid_rbf, parametros_grid_poly, parametros_grid_sigmoid]

        # Crear un objeto KFold para la validación cruzada
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        # Crear el objeto GridSearchCV
        grid_search = GridSearchCV(
            estimator=modelo_svm,
            param_grid=parametros_grid,
            scoring='accuracy',
            cv=kfold,
            n_jobs=-1,
            verbose=2
        )

        # Entrenar el modelo con búsqueda exhaustiva y validación cruzada
        grid_search.fit(X_train_scaled, y_train)

        # Obtener el mejor modelo y sus parámetros
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        # Predecir las etiquetas en el conjunto de prueba usando el mejor modelo
        y_pred = best_model.predict(X_test_scaled)

        # Calcular la precisión del modelo
        accuracy = accuracy_score(y_test, y_pred)

        # Imprimir los mejores parámetros y la precisión
        print(f"Mejores parámetros: {best_params}")
        print(f"Precisión del modelo SVM con características HOG: {accuracy}")

        final_time = time.time()

        execution_time = final_time - initial_time

        print(f"Tiempo de ejecución: {execution_time} segundos")

        # Hacemos Refit con los mejores parámetros
        best_svm = SVC(**best_params)
        best_svm.fit(X_train_scaled, y_train)

        y_pred = best_svm.predict(X_test_scaled)

        return accuracy_score(y_test, y_pred)
