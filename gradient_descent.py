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

    alpha = .05
    epoch = 1000

    grad_x.append(theta_0)
    grad_y.append(theta_1)
    grad_z.append(func(theta_0, theta_1))

    # Run the gradient
    for i in range(epoch):
        current_theta_0 = theta_0 - alpha * dx(theta_0, theta_1)
        current_theta_1 = theta_1 - alpha * dy(theta_0, theta_1)
        grad_x.append(current_theta_0)
        grad_y.append(current_theta_1)
        grad_z.append(func(current_theta_0, current_theta_1))
        if i%100 == 0:
            print("At: ",i ,theta_0, theta_1, grad_z[-1])

    # Update
        theta_0 = current_theta_0
        theta_1 = current_theta_1
        
        # Return last values
    return theta_0, theta_1, grad_z[-1]

theta_0, theta_1, grad_z = gradient_descent()
print("Final: ", theta_0, theta_1, grad_z)