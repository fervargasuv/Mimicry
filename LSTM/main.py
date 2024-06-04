import numpy as np
from keras.models import Sequential
from keras.layers import TimeDistributed, Conv1D, LSTM, Dropout, MaxPooling1D, Flatten, Dense
from keras.layers import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adamax, RMSprop, SGD, Adagrad, Adadelta, Nadam
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot
import keras as keras



# Define el modelo CNN-LSTM
def define_model(n_length,n_features, output_shape, optimizador):
    model = Sequential()
    model.add(keras.Input(shape=(None,n_length,n_features)))
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(100))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(output_shape, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizador, metrics=['accuracy'])
    return model

# Función para evaluar el modelo con validación cruzada
def evaluate_model_cv(trainX, trainy, optimizador,epochs,batch_size,n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    max_acurracy= 0
    for train_index, val_index in skf.split(trainX, trainy): 
        X_train_fold, X_val_fold = trainX[train_index], trainX[val_index]
        y_train_fold, y_val_fold = trainy[train_index], trainy[val_index]
        n_timesteps, n_features, n_outputs = X_train_fold.shape[1], X_train_fold.shape[2], y_train_fold.shape[1]
        n_steps, n_length = 4, 50
        X_train_fold = X_train_fold.reshape((X_train_fold.shape[0], n_steps, n_length, n_features))
        X_val_fold = X_val_fold.reshape((X_val_fold.shape[0], n_steps, n_length, n_features))

        model = define_model(n_length,n_features, n_outputs, optimizador)
        history = model.fit(X_train_fold, y_train_fold, epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(X_val_fold, y_val_fold))
        
        _, accuracy = model.evaluate(X_val_fold, y_val_fold, batch_size=batch_size, verbose=0)
        if accuracy > max_acurracy:
            max_acurracy=accuracy
            best_train_X,best_train_Y=X_train_fold,y_train_fold
            best_val_X,best_val_Y=X_val_fold,y_val_fold
            best_model=model

    return max_acurracy,best_model,best_train_X,best_train_Y,best_val_X,best_val_Y

# Función principal para ejecutar el experimento con Random Search y validación cruzada
def run_random_search_cv(trainX, trainy, optimizadores, epochs_range, batch_sizes_range, n_configs=1):
    best_accuracy = 0.0
    best_config = None
    best_Model=None
    for i in range(n_configs):
        epochs = np.random.randint(epochs_range[0], epochs_range[1] + 1)
        batch_size = np.random.choice(batch_sizes_range)
        
        accuracy,best_model,best_train_X,best_train_Y,best_val_X,best_val_Y = evaluate_model_cv(trainX, trainy, optimizadores,epochs,batch_size)
        # print(f'Config {i+1}: epochs={epochs}, batch_size={batch_size}, accuracy={accuracy}')
        
        if accuracy > best_accuracy:
            trainX_f,trainY_f,valX_f,valy_f=best_train_X,best_train_Y,best_val_X,best_val_Y
            best_accuracy = accuracy
            best_config = {'epochs': epochs, 'batch_size': batch_size}
            best_Model=best_model
            print("Mejor modelo encontrado:")
            print(f'Config: epochs={epochs}, batch_size={batch_size}, accuracy={accuracy}')

    # print(f'\nMejor configuración encontrada: epochs={best_config["epochs"]}, batch_size={best_config["batch_size"]}, accuracy={best_accuracy}')
    return best_config, best_accuracy,best_Model,trainX_f,trainY_f,valX_f,valy_f

# Definir los rangos de búsqueda para epochs y batch_size
epochs_range = (1, 200)
batch_sizes_range = [32,64,128]
optimizador = 'adam'

# Cargar los datos desde los archivos .npy
X_train = np.load('C:/Users/nekos/OneDrive/Documentos/Tesis/Codigo tesis/Procesamiento_datos/TrainData/new_X_train.npy')
Y_train = np.load('C:/Users/nekos/OneDrive/Documentos/Tesis/Codigo tesis/Procesamiento_datos/TrainData/new_Y_train.npy')
# Ejecutar la búsqueda aleatoria con validación cruzada
best_config, best_accuracy,best_model,trainX_f,trainY_f,valX_f,valy_f = run_random_search_cv(X_train, Y_train, optimizador, epochs_range, batch_sizes_range)



# print("La mejor acurracy es", best_accuracy)
# print("La mejor configuración es", best_config)
best_model.save('modelos/mejor_modelo_kfolds_.keras')

np.save('TrainTestFolds/trainX_folds',trainX_f)
np.save('TrainTestFolds/trainY_folds',trainY_f)
np.save('TrainTestFolds/valX_folds',valX_f)
np.save('TrainTestFolds/valY_folds',valy_f)