import tensorflow as tf

class PhysicsInformedNN(tf.Module):

    # Constructor de la clase
    def __init__(self, X_u, u, X_f, rhs, layers, epochs, lr):
        super().__init__()

        # Conversión de datos de entrada a tensores de TensorFlow
        #--------------------------------------------------------
        # Condiciones de frontera tipo Dirichlet:
        # Puntos (x, y) y sus valores u(x,y)
        self.xu = tf.convert_to_tensor(X_u[:, 0:1], dtype=tf.float32)
        self.yu = tf.convert_to_tensor(X_u[:, 1:2], dtype=tf.float32)
        self.u  = tf.convert_to_tensor(u, dtype=tf.float32)

        # Puntos de colocación internos (x, y) y el lado derecho de la EDP
        self.xf = tf.convert_to_tensor(X_f[:, 0:1], dtype=tf.float32)
        self.yf = tf.convert_to_tensor(X_f[:, 1:2], dtype=tf.float32)
        self.rhs = tf.convert_to_tensor(rhs, dtype=tf.float32)

        # Número de épocas
        self.epochs = epochs

        # Arquitectura de la red neuronal (lista de tamaños de capa)
        self.layers = layers

        # Inicialización de pesos y sesgos; creando el modelo
        self.model = self.create_model(layers)

        # Se define una política de decaimiento exponencial para la tasa de aprendizaje
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=lr,            # Tasa de aprendizaje inicial
                        decay_steps=1000,                    # Número de pasos antes de aplicar el decaimiento
                        decay_rate=0.9                       # Factor de decaimiento
                        )

        # Optimizador (Adam)
        self.optimizer = tf.keras.optimizers.Adam()

    #-------------------------------
    def create_model(self, layers):
      structure = []
      for N in layers:
        structure.append(tf.keras.layers.Dense(N,
                                               activation = 'tanh',
                                               kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                               bias_initializer='zeros'))
      model = tf.keras.Sequential(structure)
      return model

    #-------------------------------
    # Evaluación de la función
    def net_u(self, x, y):
        X = tf.concat([x, y], axis=1)  # Concatenación de coordenadas en un solo tensor
        return self.model(X)

    #-------------------------------
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
    # Función de pérdida total
    def loss_fn(self):
        u_pred = self.net_u(self.xu, self.yu)                   # Predicción de u en condiciones de frontera
        f_pred = self.net_f(self.xf, self.yf)                   # Predicción del residuo de la EDP

        loss_u = tf.reduce_mean(tf.square(self.u   - u_pred))   # Error en condiciones de frontera
        loss_f = tf.reduce_mean(tf.square(self.rhs - f_pred))   # Residuo de la EDP
        return loss_u + loss_f                                  # Pérdida total combinada

    #-------------------------------
    # Paso de entrenamiento optimizado
    @tf.function
    def train_step(self):
      with tf.GradientTape() as tape:                   # Graba operaciones para calcular gradientes
          loss_value = self.loss_fn()                   # Evalúa la función de pérdida

      variables = self.model.trainable_variables             # Obtiene las variables (pesos y sesgos) entrenables del modelo
      grads     = tape.gradient(loss_value, variables)       # Calcula los gradientes de la pérdida respecto a las variables
      self.optimizer.apply_gradients(zip(grads, variables))  # Aplica los gradientes usando el optimizador

      return loss_value

    #-------------------------------
    # Proceso de entrenamiento
    def train(self):
        for epoch in range(self.epochs):
            loss = self.train_step()  # Ejecuta un paso de entrenamiento
            if epoch % 100 == 0:      # Imprime cada 100 épocas
                print(f"Epoch {epoch}: Loss = {loss.numpy():.5e}")

    #-------------------------------
    # Predice u(x,y) para nuevos puntos
    def predict(self, X_star):
        x_star = tf.convert_to_tensor(X_star[:, 0:1], dtype=tf.float32)
        y_star = tf.convert_to_tensor(X_star[:, 1:2], dtype=tf.float32)
        return self.net_u(x_star, y_star).numpy()