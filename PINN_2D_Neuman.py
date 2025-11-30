import tensorflow as tf

class PhysicsInformedNN:

    # Constructor de la clase
    def __init__(self, X_u, u, X_n, un, X_f, rhs, layers, epochs, lr, rate_decay = True):

        """
        Información del programa, se explican las variables utilizadas
        X_u       : Coordenadas de los puntos de frontera Dirichlet,                       (N_u,  2)
        u         : Valores de Dirichlet de la frontera,                                   (N_u,  1)
        X_n       : Coordenadas de los puntos y direcciones normales de frontera Neumann,  (N_un, 2)
        un        : Valores de Neumann en la frontera,                                     (N_un, 1)
        X_f       : Puntos del interior                                                    (Np,   2)
        rhs       : Valores del lado derecho en X_f                                        (Np,   1)
        layers    : Estructura de la NN
        epochs    : Número de épocas para entrenamiento
        lr        : Tasa de aprendizaje inicial
        rate_decay: Para activar el uso de lr_schedule
        """

        # Conversión de datos de entrada a tensores de TensorFlow
        #-------------------------------
        # Condiciones de frontera tipo Dirichlet:
        # Puntos (x, y) y sus valores u(x,y)
        self.xu = tf.convert_to_tensor(X_u[:, 0:1], dtype=tf.float32)
        self.yu = tf.convert_to_tensor(X_u[:, 1:2], dtype=tf.float32)
        self.u = tf.convert_to_tensor(u, dtype=tf.float32)
        #-------------------------------
        # Condiciones de frontera tipo Neumann: Se terminan los vectores afuera.
        # Puntos (x, y), componentes normales (n_x, n_y) y derivada un = ∂u/∂n
        self.xn = tf.convert_to_tensor(X_n[:, 0:1], dtype=tf.float32)
        self.yn = tf.convert_to_tensor(X_n[:, 1:2], dtype=tf.float32)
        self.nx = tf.convert_to_tensor(X_n[:, 2:3], dtype=tf.float32)
        self.ny = tf.convert_to_tensor(X_n[:, 3:4], dtype=tf.float32)
        self.un = tf.convert_to_tensor(un, dtype=tf.float32)
        #-------------------------------
        # Puntos de colocalización internos (x, y) para evaluar el residuo de la EDP
        self.xf = tf.convert_to_tensor(X_f[:, 0:1], dtype=tf.float32) # Puntos de colocación coor x
        self.yf = tf.convert_to_tensor(X_f[:, 1:2], dtype=tf.float32) # Puntos de colocación coor y
        self.rhs = tf.convert_to_tensor(rhs, dtype=tf.float32)        # Valores de la función

        # Construcción de la red neuronal (arquitectura definida)
        self.model = self.build_model(layers)
        self.loss_history = []    # Registro de pérdida por época
        self.u_loss_history = []  # Registro de pérdida en Dirichlet x época
        self.un_loss_history = [] # Registro de pérdida en Neumann x época
        self.f_loss_history = []  # Registro de pérdida de colocacción x época

        # Número de épocas
        self.epochs = epochs

        # Tasa de aprendizaje
        self.lr = lr

        # Usar Rate_decay
        self.rate_decay = rate_decay


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


    # Definición de funciones auxiliares para la solución y derivadas


    #-------------------------------
    # Red neuronal que aproxima u(x,y)
    def net_u(self, x, y):
        X = tf.concat([x, y], axis=1)
        return self.model(X)


    #-------------------------------
    # Derivada ∂u/∂x
    def net_ux(self, x, y):
        with tf.GradientTape() as tape:
            tape.watch(x)
            u = self.net_u(x, y)
        return tape.gradient(u, x)


    #-------------------------------
    # Derivada ∂u/∂y
    def net_uy(self, x, y):

        with tf.GradientTape() as tape:
            tape.watch(y)
            u = self.net_u(x, y)
        return tape.gradient(u, y)

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
        ux_pred = self.net_ux(self.xn, self.yn)  # Predicción de u en c.f. Neumann
        uy_pred = self.net_uy(self.xn, self.yn)  # Predicción de u en c.f. Neumann
        f_pred  = self.net_f(self.xf, self.yf)   # Predicción del residuo de la EDP

        loss_u  = tf.reduce_mean(tf.square(self.u - u_pred))                                    # Error en condiciones Dirichlet
        loss_un = tf.reduce_mean(tf.square(self.un - (ux_pred * self.nx + uy_pred * self.ny)))  # Error en condiciones Neumann
        loss_f  = tf.reduce_mean(tf.square(self.rhs - f_pred))                                  # Residuo de la EDP
        return loss_u, loss_un, loss_f, loss_u + loss_un + loss_f                               # Pérdida total combinada


    @tf.function
    #-------------------------------
    # Paso de entrenamiento (una iteración de optimización)
    def train_step(self, optimizer):
        with tf.GradientTape() as tape:
            u, un, f, loss  = self.loss_fn()                                        # Calcula la pérdida
        gradients = tape.gradient(loss, self.model.trainable_variables)   # Calcula gradientes
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return u, un, f, loss

    #-------------------------------
    # Entrenamiento completo usando optimizador Adam
    def train(self):
        if self.rate_decay == True:
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                            initial_learning_rate= self.lr,
                            decay_steps= 1000,
                            decay_rate= 0.9,
                            staircase=True   # decay por saltos (no continuo)
                            )
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        for epoch in range(self.epochs):
            u, un, f, loss = self.train_step(optimizer)
            self.loss_history.append(loss.numpy())
            self.u_loss_history.append(u.numpy())
            self.un_loss_history.append(un.numpy())
            self.f_loss_history.append(f.numpy())
            
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss.numpy():.5e},  U_loss: {u.numpy():.5e}, Un_loss: {un.numpy():.5e}, f_loss: {f.numpy():.5e}')


    #-------------------------------
    # Predice u(x,y) para nuevos puntos
    def predict(self, X_star):
        x = tf.convert_to_tensor(X_star[:, 0:1], dtype=tf.float32)
        y = tf.convert_to_tensor(X_star[:, 1:2], dtype=tf.float32)
        return self.net_u(x, y).numpy()

    #-------------------------------
    def history(self):
        return self.loss_history, self.u_loss_history, self.un_loss_history, self.f_loss_history