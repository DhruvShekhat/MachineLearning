import numpy as np 

def gradient_descent(x,y):
    m_cu = b_cu = 0
    itera = 10000
    n = len(x)
    lr = 0.08
    for i in range(itera):
        y_pred = m_cu * x + b_cu
        cost = (1/n)*sum([val**2 for val in (y-y_pred)])
        m_der = -(2/n)*sum(x*(y-y_pred))
        b_der = -(2/n)*sum(y-y_pred)
        m_cu = m_cu - lr*m_der
        b_cu = b_cu - lr*b_der
        print("m:{},b:{},cost:{},iteration:{}".format(m_cu,b_cu,cost,i))


x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

gradient_descent(x,y)