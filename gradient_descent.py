import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf

# Create gradient arrays
grad_x = [] 
grad_y = []
grad_z = []
alpha = .01
epoch = 1000

def dx (x, y):
    return 8*x - 2*y
def dy (x, y):
    return 4*y - 2*x
def func(x, y):
    return 4*x*x + 2*y*y - 2*x*y

def plot_linear_data(p_x, p_y, p_z):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(-3.0, 30.0, 0.05)
    X, Y = np.meshgrid(x, y)
    zs = np.array([func(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax.plot3D(p_x, p_y, p_z, 'red')
    plt.show()

def gradient_descent():
    # Our initinal guess
    theta_0  = 25
    theta_1  = 35

    grad_x.append(theta_0)
    grad_y.append(theta_1)
    grad_z.append(func(theta_0, theta_1))

    # Run the gradient
    for i in range(epoch):
        if i%100 == 0:
            print("Iteration %d: X= %f Y= %f Z= %f" % (i ,theta_0, theta_1, grad_z[-1]))
        current_theta_0 = theta_0 - alpha * dx(theta_0, theta_1)
        current_theta_1 = theta_1 - alpha * dy(theta_0, theta_1)
        grad_x.append(current_theta_0)
        grad_y.append(current_theta_1)
        grad_z.append(func(current_theta_0, current_theta_1))
        
        # Update
        theta_0 = current_theta_0
        theta_1 = current_theta_1
        
        # Return last values
    return theta_0, theta_1, grad_z[-1]

def gradient_descent_tensorflow():    
    x = tf.Variable(25, dtype=tf.float32)
    y = tf.Variable(35, dtype=tf.float32)

    cost = 4*x*x + 2*y*y - 2*x*y
    train = tf.train.GradientDescentOptimizer(alpha).minimize(cost)

    init = tf.global_variables_initializer()
    session = tf.Session()
    session.run(init)

    for i in range(epoch):
      if i%100 == 0:
         print("Iteration %d: X= %f Y= %f Z= %f" % (i ,session.run(x), session.run(y), session.run(cost)))
      session.run(train)

print("\n\n--------------- Gradient Descent with Python -----------------------------")
theta_0, theta_1, grad_z = gradient_descent()
print("Final Gradient Descent: X= %f Y= %f Z= %f" % (theta_0, theta_1, grad_z))
plot_linear_data(grad_x, grad_y, grad_z)

print("\n\n--------------- Gradient Descent with Tensorflow -----------------------------")
gradient_descent_tensorflow()