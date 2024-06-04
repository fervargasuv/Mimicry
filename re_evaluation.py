from keras.models import load_model
from matplotlib import pyplot
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def prediccion_y_evaluacion_modelo(modelo,Xtest,Ytest,n_steps,n_length,threshold):
    # Cargar el modelo guardado
    loaded_model = load_model('mejor_modelo_kfolds_0767_1251_128.keras')

    # Cargar los datos de test de los archivos .npy
    X_test= np.load(Xtest)
    Y_test= np.load(Ytest)
    max_acurracy=0
    #Ajuste de los datos a la forma necesaria para el modelo
    X_test_adjusted = X_test.reshape((X_test.shape[0], n_steps, n_length, X_test.shape[2]))

    # Obtener las predicciones del modelo
    predictions = loaded_model.predict(X_test_adjusted)

    # Convertir las predicciones a clases (0 o 1) usando un umbral (ejemplo: 0.5)
    predicted_classes = (predictions > threshold).astype(int)

    # Obtener las clases reales
    true_classes = Y_test

    # Calcular el F1-score
    f1score = f1_score(true_classes, predicted_classes)
    # Calcular la precisión y el recall
    precision = precision_score(true_classes, predicted_classes)
    recall = recall_score(true_classes, predicted_classes)
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1-score: {f1score}')
    
    return precision,recall,f1score,predicted_classes,true_classes


precision,recall,f1score,predicted,testing=prediccion_y_evaluacion_modelo('mejor_modelo_kfolds_0767_1251_128.keras','X_test_1.npy','Y_test_1.npy',4,50,0.5)


# if precision > 0.65:
#     if precision > max_acurracy:
#         max_acurracy=precision
#         print(f'Accuracy after retraining: {precision}')
#         loaded_model.save('mejor_modelo_reentrenado_kfolds_.keras')

# #3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step
# Precision: 0.7083333333333334
# Recall: 0.8947368421052632
# F1-score: 0.7906976744186046
#mejor_modelo_kfolds_0767_1251_128
#mejor_modelo_reentrenado_kfolds_0708