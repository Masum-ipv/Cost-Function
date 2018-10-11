import numpy as np
#import tensorflow as tf

def dx (x, y):
    return 8*x - 2*y
def dy (x, y):
    return 4*y - 2*x
def func(x, y):
    return 4*x*x + 2*y*y - 2*x*y

def gradient_descent():
    # Create gradient arrays
    grad_x = [] 
    grad_y = []
    grad_z = []

    # Our initinal guess
    theta_0  = 25
    theta_1  = 35

    alpha = .01
    epoch = 1000

    grad_x.append(theta_0)
    grad_y.append(theta_1)
    grad_z.append(func(theta_0, theta_1))

    # Run the gradient
    for i in range(epoch):
        if i%100 == 0:
            print("At: ",i ,theta_0, theta_1, grad_z[-1])
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
    #cost = x**2 - 10*x + 25
    train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    init = tf.global_variables_initializer()
    session = tf.Session()
    session.run(init)

    for i in range(1000):
      if i%100 == 0:
         print("At: ",i ,session.run(x), session.run(y), session.run(cost))
      session.run(train)
      
    print(session.run(x))
    print(session.run(y))
    print(session.run(cost))

theta_0, theta_1, grad_z = gradient_descent()
print("Final: ", theta_0, theta_1, grad_z)

gradient_descent_tensorflow()