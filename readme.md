Run the code with
```python test.py``` (no dependencies)

Current output will be:
```
FORWARD
Operation: Multiply Input: [[1, -1], [1, 1, 1, -1, -1, -1]] Output: [2, 2, 2] 

Operation: Sigmoid Input: [[2, 2, 2]] Output: [0.8807970779778823, 0.8807970779778823, 0.8807970779778823] 

Operation: Multiply Input: [[0.8807970779778823, 0.8807970779778823, 0.8807970779778823], [1, 1, -1, -1, -1, -1]] Output: [-0.8807970779778823, -0.8807970779778823] 

Operation: SoftMax Input: [[-0.8807970779778823, -0.8807970779778823]] Output: [0.5, 0.5] 



BACKWARD
###############
Operation: SoftMax
Gradient of layer after this: [1, 0]

Input: After weights 2 -> [-0.8807970779778823, -0.8807970779778823]
Gradient: [-0.5, 0.5]
###############

###############
Operation: Multiply
Gradient of layer after this: [-0.5, 0.5]

Input: Sigmoid Applied -> [0.8807970779778823, 0.8807970779778823, 0.8807970779778823]
Gradient: [0.0, 0.0, -1.7615941559557646]

Input: Second layer weights -> [1, 1, -1, -1, -1, -1]
Gradient: [-0.44039853898894116, -0.44039853898894116, -0.44039853898894116, -0.44039853898894116, -0.44039853898894116, -0.44039853898894116]
###############

###############
Operation: Sigmoid
Gradient of layer after this: [0.0, 0.0, -1.7615941559557646]

Input: Inputs multiplied by weights -> [2, 2, 2]
Gradient: [0.0, 0.0, -0.18495608645965975]
###############

###############
Operation: Multiply
Gradient of layer after this: [0.0, 0.0, -0.18495608645965975]

Input: Network inputs -> [1, -1]
Gradient: [3, 3]

Input: First layer weights -> [1, 1, 1, -1, -1, -1]
Gradient: [0.0, 0.0, 0.0, -0.0, -0.0, -0.0]
###############
```