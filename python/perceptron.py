def perceptron(inputs, weights, bias):
    outputs = []
    for i in range(len(weights[0])):
        sumPe = 0
        for j in range(len(inputs)):
            sumPe += inputs[j] * weights[j][i]
        sumPe += bias[i]
        outputs.append(sumPe)   
    return outputs
