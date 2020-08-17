import time
import numpy as np
import matplotlib.pyplot as plt

class Data(object):

    def __init__(self, path):
        self.dataset = self._build_dataset(path)

    def _build_dataset(self, path):
        structure = [('x', np.float),
                     ('y', np.float)]

        with open(path, encoding="utf8") as data_csv:
            data_gen = ((float(line.split(',')[0]), float(line.split(',')[1]))
                        for i, line in enumerate(data_csv) if i != 0)
            embeddings = np.fromiter(data_gen, structure)

        return embeddings

    def split(self, percentage):
        X = self.dataset[['x']]
        y = self.dataset['y']

        permuted_idxs = np.random.permutation(len(X))

        train_idxs = permuted_idxs[0:int(percentage * X.shape[0])]

        test_idxs = permuted_idxs[int(percentage * X.shape[0]): X.shape[0]]

        X_train = X[train_idxs]
        X_test = X[test_idxs]

        y_train = y[train_idxs]
        y_test = y[test_idxs]

        return X_train, X_test, y_train, y_test
        

class BaseModel(object):

    def __init__(self):
        self.model = None

    def fit(self, X, Y):
        return NotImplemented

    def predict(self, X):
        return NotImplemented


class LinearRegression(BaseModel):

    def fit(self, X, y):
        if len(X.shape) == 1:
            W = X.T.dot(y) / X.T.dot(X)
        else:
            W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        self.model = W

    def predict(self, X):
        return self.model * X


# función que implementa k-folds para hacer el fit del dataset con polonomio de grado n=1
# entrega como salida los parámetros W de todos los modelos calculados con su correspondiente MSE
def k_folds(X_train, y_train, k=5):
    l_regression = LinearRegression()
    error = MSE()
    
    chunk_size = int(X_train.shape[0] / k)
    mse_list = []
    w_array = np.zeros((1,k+1))
    j = 0
    for i in range(0, X_train.shape[0], chunk_size):
        end = i + chunk_size if i + chunk_size <= X_train.shape[0] else X_train.shape[0]
        new_X_valid = X_train[i: end]
        new_y_valid = y_train[i: end]
        new_X_train = np.concatenate([X_train[: i], X_train[end:]])
        new_y_train = np.concatenate([y_train[: i], y_train[end:]])

        l_regression.fit(new_X_train, new_y_train)
        w_array[:, j] = l_regression.model.reshape(-1)
        j = j+1
        prediction = l_regression.predict(new_X_valid)
        mse_list.append(error(new_y_valid, prediction))

    return mse_list, w_array


# función que implementa k-folds para hacer el fit del dataset con polonomio de grado n>1
# entrega como salida los parámetros W de todos los modelos calculados con su correspondiente MSE
def k_folds_n(X_train, y_train, k=5):
    l_regression = LinearRegression()
    error = MSE()
    
    chunk_size = int(X_train.shape[0] / k)
    mse_list = []
    w_array = np.zeros((X_train.shape[1],k+1))
    j = 0
    for i in range(0, X_train.shape[0], chunk_size):
        end = i + chunk_size if i + chunk_size <= X_train.shape[0] else X_train.shape[0]
        new_X_valid = X_train[i: end]
        new_y_valid = y_train[i: end]
        new_X_train = np.concatenate([X_train[: i], X_train[end:]])
        new_y_train = np.concatenate([y_train[: i], y_train[end:]])

        l_regression.fit(new_X_train, new_y_train)
        w_array[:, j] = l_regression.model.reshape(-1)
        j = j+1
        prediction = np.sum(l_regression.predict(new_X_valid), axis=1)
        mse_list.append(error(new_y_valid, prediction))

    return mse_list, w_array


# funcion que selecciona el modelo cuyo MSE sea el más próximo al valor promedio de los todos los modelos calculados
def selector(mse_list):
    mean_MSE = np.mean(mse_list)
    selected_model = np.argsort(abs(mse_list - mean_MSE))
    return selected_model[0]
    
    
class Metric(object):
    def __call__(self, target, prediction):
        return NotImplemented


class MSE(Metric):
    def __init__(self):
        Metric.__init__(self)

    def __call__(self, target, prediction):
        n = target.size
        return np.sum((target - prediction) ** 2) / n
    

# función para cálculo del gradiente minibatch para polonomio de grado n=1
# en cada epoch, calcula el error de entrenamiento y el de validación
# devuelve como salida eñ parámetro W y los valores de error de entrenamiento y de validación calculados en cada epoch
def mini_batch_gradient_descent(X, y, lr=0.01, amt_epochs=100):
    """
    shapes:
        X_t = nxm
        y_t = nx1
        W = mx1
    """
    X = X.reshape(-1,1)
    
    b = 16
    n = X.shape[0]
    m = X.shape[1]

    # initialize random weights
    W = np.random.randn(m).reshape(m, 1)
    
    train_error_vector = np.zeros(amt_epochs)
    valid_error_vector = np.zeros(amt_epochs)

    for j in range(amt_epochs): 
        permuted_idxs = np.random.permutation(len(X))
        percentage = 0.8
        train_idxs = permuted_idxs[0:int(percentage * X.shape[0])]
        valid_idxs = permuted_idxs[int(percentage * X.shape[0]): X.shape[0]]
        
        X_train = X[train_idxs]
        y_train = y[train_idxs]

        X_valid = X[valid_idxs]
        y_valid = y[valid_idxs]

        batch_size = int(len(X_train) / b)
        
        train_error = np.zeros((X_train.shape[0],1))
        
        for i in range(0, len(X_train), batch_size):
            end = i + batch_size if i + batch_size <= len(X_train) else len(X_train)
            batch_X = X_train[i: end]
            batch_y = y_train[i: end].reshape(-1,1)

            prediction = np.matmul(batch_X, W)  # nx1
            train_error_batch = batch_y - prediction  # nx1

            grad_sum = np.sum(train_error_batch * batch_X, axis=0)
            grad_mul = -2/n * grad_sum  # 1xm
            gradient = np.transpose(grad_mul).reshape(-1, 1)  # mx1

            W = W - (lr * gradient)
            
            train_error[i:end, :] = train_error_batch
        
        train_error_vector[j] = np.sum(train_error, axis = 0)
        valid_error = y_valid.reshape(-1, 1) - X_valid.dot(W).reshape(-1, 1)
        valid_error_vector[j] = np.sum(valid_error, axis = 0)

    return W, train_error_vector, valid_error_vector


# función para cálculo del gradiente minibatch para polonomio de grado n>1
# en cada epoch, calcula el error de entrenamiento y el de validación
# devuelve como salida eñ parámetro W y los valores de error de entrenamiento y de validación calculados en cada epoch
def mini_batch_gradient_descent_n(X_train, y_train, lr=0.01, amt_epochs=100):
    """
    shapes:
        X_t = nxm
        y_t = nx1
        W = mx1
    """
    b = 16
    n = X_train.shape[0]
    m = X_train.shape[1]

    # initialize random weights
    W = np.random.randn(m).reshape(m, 1)

    for i in range(amt_epochs):
        idx = np.random.permutation(X_train.shape[0])
        X_train = X_train[idx]
        y_train = y_train[idx]

        batch_size = int(len(X_train) / b)
        for i in range(0, len(X_train), batch_size):
            end = i + batch_size if i + batch_size <= len(X_train) else len(X_train)
            batch_X = X_train[i: end]
            batch_y = y_train[i: end].reshape(-1,1)

            prediction = np.matmul(batch_X, W)  # nx1
            error = batch_y - prediction  # nx1

            grad_sum = np.sum(error * batch_X, axis=0)
            grad_mul = -2/n * grad_sum  # 1xm
            gradient = np.transpose(grad_mul).reshape(-1, 1)  # mx1

            W = W - (lr * gradient)

    return W
    

# función para cálculo del gradiente minibatch para polonomio de grado n=1
# en cada epoch, calcula el error de entrenamiento y el de validación
# devuelve como salida eñ parámetro W y los valores de error de entrenamiento y de validación calculados en cada epoch
def regularization_ridge(X, y, lr=0.01, amt_epochs=100):
    """
    shapes:
        X_t = nxm
        y_t = nx1
        W = mx1
    """
    X = X.reshape(-1,1)
    
    b = 16
    n = X.shape[0]
    m = X.shape[1]

    # initialize random weights
    W = np.random.randn(m).reshape(m, 1)
    
    train_error_vector = np.zeros(amt_epochs)
    valid_error_vector = np.zeros(amt_epochs)

    for j in range(amt_epochs): 
        permuted_idxs = np.random.permutation(len(X))
        percentage = 0.8
        train_idxs = permuted_idxs[0:int(percentage * X.shape[0])]
        valid_idxs = permuted_idxs[int(percentage * X.shape[0]): X.shape[0]]
        
        X_train = X[train_idxs]
        y_train = y[train_idxs]

        X_valid = X[valid_idxs]
        y_valid = y[valid_idxs]

        batch_size = int(len(X_train) / b)
        
        train_error = np.zeros((X_train.shape[0],1))
        
        for i in range(0, len(X_train), batch_size):
            end = i + batch_size if i + batch_size <= len(X_train) else len(X_train)
            batch_X = X_train[i: end]
            batch_y = y_train[i: end].reshape(-1,1)

            prediction = np.matmul(batch_X, W)  # nx1
            train_error_batch = batch_y - prediction  # nx1

            grad_sum = np.sum(train_error_batch * batch_X, axis=0)
            grad_mul = -2/n * grad_sum - 2 * lr * W  # mx1
            gradient = np.transpose(grad_mul).reshape(-1, 1)  # mx1

            W = W - (lr * gradient)
            
            train_error[i:end, :] = train_error_batch
        
        train_error_vector[j] = np.sum(train_error, axis = 0)
        valid_error = y_valid.reshape(-1, 1) - X_valid.dot(W).reshape(-1, 1)
        valid_error_vector[j] = np.sum(valid_error, axis = 0)

    return W, train_error_vector, valid_error_vector

#---------------------------------------------------------------------------------
#%%
if __name__ == '__main__':
    # loading dataset
    dataset = Data('clase_8_dataset.csv')
    print('\nDataset loaded')
       
    X_train, X_test, y_train, y_test = dataset.split(0.8)
    print('Dataset split')
    
    X_train = np.array(X_train, dtype=float)
    X_test = np.array(X_test, dtype=float)
    
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    # plots
    fig = plt.figure()
    plt.plot(X, y, 'o', color='blue')
    plt.title('Dataset')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
    fig.savefig('dataset_class8.png')
    
    # linear regression
    print('\nRegresion lineal usando k-flods')
    error = MSE()
    error_vector = np.zeros(4)

    # n = 1
    mse_list, w_array = k_folds(X_train, y_train, k=5)
    index_w = selector(mse_list)
    w_n1 = w_array[0,index_w]
    print('\nPolinomio de grado 1:')
    print('W óptimo: ')
    print(w_n1)
    y_linear = w_n1 * X
    error_vector[0] = error(y, y_linear)

    # n = 2
    X2 = np.vstack((np.power(X_train, 2),X_train)).T
    mse_list, w_array = k_folds_n(X2, y_train, k=5)
    index_w = selector(mse_list)
    w_n2 = w_array[:,index_w]
    print('\nPolinomio de grado 2:')
    print('W óptimo: ')
    print(w_n2)
    y_quadratic = w_n2[0]*np.power(X, 2) + w_n2[1]*X
    error_vector[1] = error(y, y_quadratic)

    # n = 3
    X3 = np.vstack((np.power(X_train, 3), np.power(X_train, 2),X_train)).T
    mse_list, w_array = k_folds_n(X3, y_train, k=5)
    index_w = selector(mse_list)
    w_n3 = w_array[:,index_w]
    print('\nPolinomio de grado 3:')
    print('W óptimo: ')
    print(w_n3)
    y_cubic = w_n3[0]*np.power(X, 3) + w_n3[1]*np.power(X, 2) + w_n3[2]*X
    error_vector[2] = error(y, y_cubic)

    # n = 4
    X4 = np.vstack((np.power(X_train, 4), np.power(X_train, 3), np.power(X_train, 2),X_train)).T
    mse_list, w_array = k_folds_n(X4, y_train, k=5)
    index_w = selector(mse_list)
    w_n4 = w_array[:,index_w]
    print('\nPolinomio de grado 4:')
    print('W óptimo: ')
    print(w_n4)
    y4 = w_n4[0]*np.power(X, 4) + w_n4[1]*np.power(X, 3) + w_n4[2]*np.power(X, 2) + w_n4[3]*X
    error_vector[3] = error(y, y4)

    # selction model
    # se selecciona el modelo con el menor MSE
    selected_model = np.argsort((error_vector))
    print('\nEl polinomio que aproxima al dataset con el menor MSE es de grado', format(selected_model[0]+1))
    
    #plots
    fig = plt.figure()
    plt.subplot(1, 1, 1)
    # original
    plt.plot(X, y, 'o')
    # linear
    plt.plot(X, y_linear, 'o')
    # quadratic
    plt.plot(X, y_quadratic, 'o')
    # cubic
    plt.plot(X, y_cubic, 'o')
    # 4 power
    plt.plot(X, y4, 'o')
    
    plt.title('Dataset class 8 - Fitting curves')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(['dataset', 'linear', 'quadratic', 'cubic', '4th power'])
    plt.show()
    fig.savefig('dataset_class8_fitting_curves.png')


    # minibatch gradient
    print('\n\nMINI BATCH GRADIENT DESCENT VS LINEAR REGRESSION (n = 1)\n')
    lr_4 = 0.0001
    amt_epochs_4 = 50000
    start = time.time()
    w_manual, train_error_vector, valid_error_vector = mini_batch_gradient_descent(X_train, y_train, lr=lr_4, amt_epochs=amt_epochs_4)
    time = time.time() - start
    print('W (Linear Regression [n=1]): {}'.format(w_n1.reshape(-1)))
    print('W (Minibatch): {}'.format(w_manual.reshape(-1)))
    y_minibatch = w_manual * X
    mse_minibatch = error(y, y_minibatch)
    
    # plots training error vs validation error
    amt_epochs_vector = np.arange(len(train_error_vector))
    
    fig = plt.figure()
    plt.plot(amt_epochs_vector, train_error_vector, 'o', color='blue', label='Training Error')
    plt.plot(amt_epochs_vector, valid_error_vector, 'o', color='red', label='Validation Error')
    plt.title('minibatch: error vs. amount of epochs')
    plt.xlabel('amount of epochs')
    plt.ylabel('error')
    plt.legend()
    plt.show()
    fig.savefig('minibatch_error_vs_amount_of_epochs.png')
    
    print('\nMSE Linear Regresion vs. Minibatch\n')
    print('MSE (Linear Regresion) = {}'.format(error_vector[0]))
    print('MSE (Minibatch) = {}'.format(mse_minibatch))
    
    
    # regularization ridge (norm L2)
    print('\n\nREGULARIZATION RIDGE\n')
    w_manual_reg, train_error_vector_reg, valid_error_vector_reg = regularization_ridge(X_train, y_train, lr=lr_4, amt_epochs=amt_epochs_4)
    print('W (Linear Regression [n=1]): {}'.format(w_n1.reshape(-1)))
    print('W (Minibatch): {}'.format(w_manual.reshape(-1)))
    print('W (Regularization Ridge): {}'.format(w_manual_reg.reshape(-1)))
    y_reg_ridge = w_manual_reg * X
    mse_reg_ridge = error(y, y_reg_ridge)

    # plots training error vs validation error
    fig = plt.figure()
    plt.plot(amt_epochs_vector, train_error_vector_reg, 'o', color='blue', label='Training Error')
    plt.plot(amt_epochs_vector, valid_error_vector_reg, 'o', color='red', label='Validation Error')
    plt.title('regularization ridge: error vs. amount of epochs')
    plt.xlabel('amount of epochs')
    plt.ylabel('error')
    plt.legend()
    plt.show()
    fig.savefig('reg_error_vs_amount_of_epochs.png')
    
    print('\nMSE Linear Regresion, MSE Minibatch, Regularization Ridge - Comparation\n')
    print('MSE (Linear Regresion) = {}'.format(error_vector[0]))
    print('MSE (Minibatch) = {}'.format(mse_minibatch))
    print('MSE (Regularization Ridge) = {}'.format(mse_reg_ridge))
