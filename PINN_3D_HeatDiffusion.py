import tensorflow as tf

class PINN_HeatDiffusion(tf.Module):

    # Constructor de la clase
    def __init__(self, X_u, u, X_n, un, X_f, rhs, layers, epochs, lr):
        """
        Información del programa, se explican las variables utilizadas
        X_u       : Coordenadas de los puntos de frontera Dirichlet,                       (N_u,  3)
        u         : Valores de Dirichlet de la frontera,                                   (N_u,  1)
        X_n       : Coordenadas de los puntos y direcciones normales de frontera Neumann,  (N_un, 6)
        un        : Valores de Neumann en la frontera,                                     (N_un, 1)
        X_f       : Puntos del interior                                                    (Np,   3)
        rhs       : Valores del lado derecho en X_f                                        (Np,   1)
        layers    : Estructura de la NN
        epochs    : Número de épocas para entrenamiento
        lr        : Tasa de aprendizaje inicial
        """
        super().__init__()

        # Conversión de datos de entrada a tensores de TensorFlow
        #--------------------------------------------------------
        # Condiciones de frontera tipo Dirichlet:
        # Puntos (t, x, y) y sus valores u(t,x,y)
        self.tu = tf.convert_to_tensor(X_u[:, 0:1], dtype=tf.float32) # Time
        self.xu = tf.convert_to_tensor(X_u[:, 1:2], dtype=tf.float32) # x
        self.yu = tf.convert_to_tensor(X_u[:, 2:3], dtype=tf.float32) # y
        self.u  = tf.convert_to_tensor(u, dtype=tf.float32)
        #-------------------------------
        # Condiciones de frontera tipo Neumann: Se terminan los vectores afuera.
        # Puntos (t, x, y), componentes normales (n_t, n_x, n_y) y derivada un = ∂u/∂n
        self.tn = tf.convert_to_tensor(X_n[:, 0:1], dtype=tf.float32)
        self.xn = tf.convert_to_tensor(X_n[:, 1:2], dtype=tf.float32)
        self.yn = tf.convert_to_tensor(X_n[:, 2:3], dtype=tf.float32)
        self.nt = tf.convert_to_tensor(X_n[:, 3:4], dtype=tf.float32)
        self.nx = tf.convert_to_tensor(X_n[:, 4:5], dtype=tf.float32)
        self.ny = tf.convert_to_tensor(X_n[:, 5:6], dtype=tf.float32)
        self.un = tf.convert_to_tensor(un, dtype=tf.float32)
        #-------------------------------
        # Puntos de colocación internos (t, x, y) y el lado derecho de la EDP
        self.tf = tf.convert_to_tensor(X_f[:, 0:1], dtype=tf.float32) # To not overwrite
        self.xf = tf.convert_to_tensor(X_f[:, 1:2], dtype=tf.float32)
        self.yf = tf.convert_to_tensor(X_f[:, 2:3], dtype=tf.float32)
        self.rhs = tf.convert_to_tensor(rhs, dtype=tf.float32)

        # Número de épocas
        self.epochs = epochs

        # Arquitectura de la red neuronal (lista de tamaños de capa)
        self.layers = layers

        # Tasa de Aprendizaje
        self.lr = lr

        # Construcción de la red neuronal (arquitectura definida)
        self.model = self.build_model(layers)
        self.loss_history = []    # Registro de pérdida por época
        self.u_loss_history = []  # Registro de pérdida en Dirichlet x época
        self.un_loss_history = [] # Registro de pérdida en Neumann x época
        self.f_loss_history = []  # Registro de pérdida de colocacción x época

    #-------------------------------
    def build_model(self, layers):
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(3,)))                          # Entrada: (x, y)

        for width in layers[1:-1]:                                     # Capas ocultas
            model.add(tf.keras.layers.Dense(width, activation='tanh',
                                            kernel_initializer='glorot_normal'))

        model.add(tf.keras.layers.Dense(layers[-1], activation=None))  # Capa de salida
        return model

    #-------------------------------
    # Evaluación de la función
    def net_u(self, t, x, y):
        X = tf.concat([t, x, y], axis=1)  # Concatenación de coordenadas en un solo tensor
        return self.model(X)

    #-------------------------------
    # Derivada ∂u/∂x
    def net_ut(self, t, x, y):
        with tf.GradientTape() as tape:
            tape.watch(t)
            u = self.net_u(t, x, y)
        return tape.gradient(u, t)
        
    #-------------------------------
    # Derivada ∂u/∂x
    def net_ux(self, t, x, y):
        with tf.GradientTape() as tape:
            tape.watch(x)
            u = self.net_u(t, x, y)
        return tape.gradient(u, x)

    #-------------------------------
    # Derivada ∂u/∂y
    def net_uy(self, t, x, y):
        with tf.GradientTape() as tape:
            tape.watch(y)
            u = self.net_u(t, x, y)
        return tape.gradient(u, y)

    #-------------------------------
    # Red neuronal para evaluar la la EDP: f(x,y) = u_xx + u_yy
    def net_f(self, t, x, y):
        with tf.GradientTape(persistent=True) as tape2:     # Segundo orden: d²u/dx² y d²u/dy²
            tape2.watch([t, x, y])
            with tf.GradientTape(persistent=True) as tape1: # Primer orden: du/dx y du/dy
                tape1.watch([t, x, y])
                u = self.net_u(t, x, y)
            u_t = tape1.gradient(u, t)     # Derivada parcial de u respecto a t
            u_x = tape1.gradient(u, x)     # Derivada parcial de u respecto a x
            u_y = tape1.gradient(u, y)     # Derivada parcial de u respecto a y
        u_xx = tape2.gradient(u_x, x)      # Segunda derivada de u respecto a x
        u_yy = tape2.gradient(u_y, y)      # Segunda derivada de u respecto a y
        del tape1                          # Libera recursos
        del tape2
        return u_t - (u_xx + u_yy)   # Heat Diffusion Eq, u_t = a(u_xx + u_yy)

    #-------------------------------
    # Función de pérdida total
    def loss_fn(self):
        u_pred = self.net_u(self.tu, self.xu, self.yu)                   # Predicción de u en condiciones de frontera
        ut_pred = self.net_ut(self.tn, self.xn, self.yn)                 # Predicción de u_t
        ux_pred = self.net_ux(self.tn, self.xn, self.yn)                 # Predicción de u_x
        uy_pred = self.net_uy(self.tn, self.xn, self.yn)                 # Predicción de u_y
        f_pred = self.net_f(self.tf, self.xf, self.yf)                   # Predicción del residuo de la EDP

        un_pred = ut_pred*self.nt + ux_pred * self.nx + uy_pred * self.ny #Predicción de c.f. Neumann

        loss_u = tf.reduce_mean(tf.square(self.u   - u_pred))            # Error en condiciones de frontera
        loss_un = tf.reduce_mean(tf.square(self.un - un_pred))           # Error en condiciones Neumann
        loss_f = tf.reduce_mean(tf.square(self.rhs - f_pred))            # Residuo de la EDP
        return loss_u, loss_un, loss_f, loss_u + loss_un + loss_f        # Pérdida total combinada

    #-------------------------------
    # Paso de entrenamiento optimizado
    @tf.function
    def train_step(self, optimizer):
      with tf.GradientTape() as tape:                   # Graba operaciones para calcular gradientes
          u, un, f, loss_value = self.loss_fn()       # Evalúa la función de pérdida

      variables = self.model.trainable_variables             # Obtiene las variables (pesos y sesgos) entrenables del modelo
      grads     = tape.gradient(loss_value, variables)  # Calcula los gradientes de la pérdida respecto a las variables
      optimizer.apply_gradients(zip(grads, variables))  # Aplica los gradientes usando el optimizador
      return u, un, f, loss_value

    #-------------------------------
    # Proceso de entrenamiento
    # Entrenamiento completo usando optimizador Adam
    def train(self):
        # To use a rate decay to minimize oscilations
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate= self.lr,
                        decay_steps= 500,
                        decay_rate= 0.9,
                        staircase=True   # decay por saltos
                        )
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
            
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
        t_star = tf.convert_to_tensor(X_star[:, 0:1], dtype=tf.float32)
        x_star = tf.convert_to_tensor(X_star[:, 1:2], dtype=tf.float32)
        y_star = tf.convert_to_tensor(X_star[:, 2:3], dtype=tf.float32)
        return self.net_u(t_star, x_star, y_star).numpy()

    #-------------------------------
    # Reporte de los errores de entrenamiento
    def history(self):
        """
        The order of histories is
        General:   loss_history
        Dirichlet: u_loss_history
        Neumann:   un_loss_history
        Interior:  f_loss_history
        """
        return self.loss_history, self.u_loss_history, self.un_loss_history, self.f_loss_history