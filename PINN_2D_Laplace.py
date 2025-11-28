import tensorflow as tf

class PhysicsInformedNN:

    # Constructor de la clase
    def __init__(self, X_u, u, X_f, rhs, layers, epochs, lr):

        # Conversión de datos de entrada a tensores de TensorFlow
        #-------------------------------
        # Condiciones de frontera tipo Dirichlet:
        # Puntos (x, y) y sus valores u(x,y)
        self.xu = tf.convert_to_tensor(X_u[:, 0:1], dtype=tf.float32)
        self.yu = tf.convert_to_tensor(X_u[:, 1:2], dtype=tf.float32)
        self.u = tf.convert_to_tensor(u, dtype=tf.float32)
        #-------------------------------
        # Puntos de colocalización internos (x, y) para evaluar el residuo de la EDP
        self.xf = tf.convert_to_tensor(X_f[:, 0:1], dtype=tf.float32) # Puntos de colocación coor x
        self.yf = tf.convert_to_tensor(X_f[:, 1:2], dtype=tf.float32) # Puntos de colocación coor y
        self.rhs = tf.convert_to_tensor(rhs, dtype=tf.float32)        # Valores de la función

        #-------------------------------

        # Construcción de la red neuronal (arquitectura definida)
        self.model = self.build_model(layers)
        self.loss_history = []  # Registro de pérdida por época
        self.u_loss_history = [] # Registro de pérdida en la frontera x época
        self.f_loss_history = [] # Registro de pérdida de colocacción x época

        # Número de épocas
        self.epochs = epochs

        # Tasa de aprendizaje
        self.lr = lr

    # Construye una red neuronal totalmente conectada con:
    #   - activación tanh
    #   - pesos inicializados con glorot
    def build_model(self, layers):
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(2,)))                          # Entrada: (x, y)

        for width in layers[1:-1]:                                     # Capas ocultas
            model.add(tf.keras.layers.Dense(width, activation='tanh',
                                            kernel_initializer='glorot_normal'))

        model.add(tf.keras.layers.Dense(layers[-1], activation=None))  # Capa de salida
        return model

    #-------------------------------
    # Red neuronal que aproxima u(x,y)
    def net_u(self, x, y):
        X = tf.concat([x, y], axis=1)
        return self.model(X)

    #------------------------------
    # Red neuronal para evaluar la la EDP: f(x,y) = u_xx + u_yy
    def net_f(self, x, y):
        with tf.GradientTape(persistent=True) as tape2:     # Segundo orden: d²u/dx² y d²u/dy²
            tape2.watch([x, y])
            with tf.GradientTape(persistent=True) as tape1: # Primer orden: du/dx y du/dy
                tape1.watch([x, y])
                u = self.net_u(x, y)
            u_x = tape1.gradient(u, x)     # Derivada parcial de u respecto a x
            u_y = tape1.gradient(u, y)     # Derivada parcial de u respecto a y
        u_xx = tape2.gradient(u_x, x)      # Segunda derivada de u respecto a x
        u_yy = tape2.gradient(u_y, y)      # Segunda derivada de u respecto a y
        del tape1                          # Libera recursos
        del tape2
        return u_xx + u_yy                 # Evalúa el operador de Laplace (u_xx + u_yy)


    #-------------------------------
    # Función de pérdida combinada:
    # - Error en condiciones Dirichlet
    # - Error en condiciones Neumann
    # - Error en el residuo de la ecuación (puntos interiores)
    def loss_fn(self):
        u_pred  = self.net_u(self.xu, self.yu)   # Predicción de u en c.f. Dirichlet
        f_pred  = self.net_f(self.xf, self.yf)   # Predicción del residuo de la EDP

        loss_u  = tf.reduce_mean(tf.square(self.u - u_pred))                                    # Error en condiciones Dirichlet
        loss_f  = tf.reduce_mean(tf.square(self.rhs - f_pred))                                  # Residuo de la EDP
        return loss_u, loss_f, loss_u + loss_f                                                  # Pérdida total combinada


    @tf.function
    #-------------------------------
    # Paso de entrenamiento (una iteración de optimización)
    def train_step(self, optimizer):
        with tf.GradientTape() as tape:
            u, f, loss  = self.loss_fn()                                        # Calcula la pérdida
        gradients = tape.gradient(loss, self.model.trainable_variables)         # Calcula gradientes
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return u, f, loss

    #-------------------------------
    # Entrenamiento completo usando optimizador Adam
    def train(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        for epoch in range(self.epochs):
            u, f, loss = self.train_step(optimizer)
            self.loss_history.append(loss.numpy())
            self.u_loss_history.append(u.numpy())
            self.f_loss_history.append(f.numpy())
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss.numpy():.5e},  U_loss: {u.numpy():.5e}, f_loss: {f.numpy():.5e}')


    #-------------------------------
    # Predice u(x,y) para nuevos puntos
    def predict(self, X_star):
        x = tf.convert_to_tensor(X_star[:, 0:1], dtype=tf.float32)
        y = tf.convert_to_tensor(X_star[:, 1:2], dtype=tf.float32)
        return self.net_u(x, y).numpy()

    #------------------------------
    def history(self):
        """
        Returns the different losses of the training:
        1) General loss
        2) Boundry loss
        3) Interior loss
        """
        return self.loss_history, self.u_loss_history, self.f_loss_history