'''
I modified the code here:
https://school.geekwall.in/p/Sy9NonT37/creating-a-neural-network-from-scratch-in-python
To make a double layer Neural Network
Check the link to see more info, and learn a example a bit more simple
'''
import numpy as np
'''
I use the sigmoid function but, but u can use another activation functions
If u change it, remember to change the derivative right bellow too
'''
def actvation_func(x):
    return 1.0/(1+ np.exp(-x))
'''
Actually the devirative of the function that i'm using (Sigmoid) is : sigmoid(x)*(1 - sigmoid(x))
But along my code i found that it causes some overflow, so i changed it
But still a convergent function as the actual derivative 
'''
def derivative_func(x):
    return x*(1 - x)
'''
I still need to implement the bias part
So just imagine that the actual process is Sum(Wi*Ai)
'''
class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weight1 = np.random.rand(self.input.shape[1],5)
        self.weight2 = np.random.rand(5,4)
        self.weight3 = np.random.rand(4,1)
        #self.bias1 = np.random.rand(5)
        #self.bias2 = np.random.rand(4)
        #self.bias3 = np.random.rand(1)
        self.expectations = y
        self.output = np.zeros(y.shape)

    def feedforward(self):
        self.layer1 =  actvation_func(np.dot(self.input, self.weight1))
        self.layer2 = actvation_func(np.dot(self.layer1, self.weight2))
        self.output = actvation_func(np.dot(self.layer2, self.weight3))

    def backprop(self):
        '''
        If u dont know how the calculation of back propagation works, check this link:
        https://www.youtube.com/watch?v=tIeHLnjs5U8
        Actually the whole playlist is amazing, it has 4 videos
        '''
        d_output = (2*(self.expectations - self.output) * derivative_func(self.output))
        #d_bias3 = d_output
        d_weight3 = np.dot(self.layer2.T, d_output)
        d_layer2 = (np.dot(d_output, self.weight3.T)*derivative_func(self.layer2))
        #d_bias2 = d_layer2
        d_weight2 = np.dot(self.layer1.T, d_layer2)
        d_layer1 = (np.dot(d_layer2, self.weight2.T)*derivative_func(self.layer1))
        #d_bias1 = d_layer1
        d_weight1 = np.dot(self.input.T, d_layer1)
    
        self.weight1 += d_weight1*0.2
        self.weight2 += d_weight2*0.2
        self.weight3 += d_weight3*0.2
        '''
        This value 0.2, is the size of step that i will move in the direction pointed by the gradient
        Try to change it and see how accurate it can gets
        '''
        #self.bias1 += d_bias1*0.5
        #self.bias2 += d_bias2*0.5
        #self.bias3 += d_bias3*0.5 

if __name__ == "__main__":
    x = np.array([[0,0,0,0,0,0],
                  [0,0,0,0,0,1],
                  [0,0,0,0,1,0],
                  [0,0,0,0,1,1],
                  [1,0,0,1,0,0],
                  [0,0,0,1,0,1],
                  [0,0,0,1,1,0],
                  [0,0,0,1,1,1],
                  [0,0,1,0,0,0],
                  [0,0,1,0,0,1],
                  [0,0,1,0,1,0],
                  [1,0,1,0,1,1]])
    y = np.array([[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1]])
    nn = NeuralNetwork(x,y)
    
    '''
    Here i take the input and the output that i had and make the neural network smart
    If u dont get i enter a binary number with ',' bet the digits
    Then the expected output can be:
                                0 if its even
                                1 if its odd
    '''
    for i in range(30000):
        nn.feedforward()
        nn.backprop()

    z = np.array([[1,1,0,0,1,0],
                  [1,1,1,1,0,1]])

    nn.input = z
    nn.feedforward()
    print("NEW INPUT REPLY")   
    for i in range(z.shape[0]):
        print("The number ", z[i] ," has ",(nn.output[i][0]*100), "% chance of being odd", sep='')

        