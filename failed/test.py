from nodes import TensorNode, OpNode
from operations import Sigmoid, Expand, Plus, RowSum, Multiply, SoftMax

# Now to run backrpopagation we simply traverse the network of nodes forward calling the forward functions (all ancestors need to have been called (otherwise incomplete input))

# And then traverse the network of nodes backward calling the backwards functions (all decendents need to have been called otherwise incomplete output)

# Input node, so no source
X = TensorNode(init_value=[1, -1], name='Network inputs')
w_1 = TensorNode(init_value=[1, 1, 1, -1, -1,-1], name='First layer weights')


# Weigh the input by a weights vector
h_1 = TensorNode(name='Inputs multiplied by weights')
weighting_1 = OpNode(op=Multiply(), inputs=[X, w_1], outputs=[h_1])

# Operation that sigmoids the input value in a single neuron
s_1 = TensorNode(name='Sigmoid Applied')
sigm = OpNode(op=Sigmoid(), inputs=[h_1], outputs=[s_1])

# Second layer of weights
s_2 = TensorNode(name='After weights 2')
w_2 = TensorNode(init_value=[1, 1, -1, -1, -1,-1], name='Second layer weights')
weighting_2 = OpNode(op=Multiply(), inputs=[s_1, w_2], outputs=[s_2])

y = TensorNode(name='After softmax')
softmax = OpNode(op=SoftMax(), inputs=[s_2], outputs=[y])

network = [weighting_1 , sigm, weighting_2, softmax] # , biasing, 

# Forward pass
print("\nFORWARD")
for operation in network:
    # For every output of this operation
    for output in operation.outputs:
        output.value = operation.op.forward(*operation.inputs).value

        print('Operation:', operation.op.__class__.__name__, 'Input:', [i.value for i in operation.inputs], 'Output:', output.value, '\n')


# Backward pass
print("\n\nBACKWARD")
last_output_node_inputs = []

for operation in network[::-1]:
    if network.index(operation)<len(network)-1:
         output_node_gradient = last_output_node_gradient
    else: output_node_gradient = [1, 0]

    args = [output_node_gradient]
    if len(last_output_node_inputs) > 1:
        args.append(last_output_node_inputs[1])
    print('###############\nOperation:', operation.op.__class__.__name__)
    print('Gradient of layer after this:', output_node_gradient)
    for i, gradient in enumerate(operation.op.backward(operation, *args)):
        operation.inputs[i].gradient = gradient # Ugly hack, but softmax gradient is [[0.25,0.25], [0.25,0.25]]

        print('\nInput:', operation.inputs[i].name,'->' ,operation.inputs[i].value)
        print('Gradient:', operation.inputs[i].gradient)
    last_output_node_gradient = [input.gradient for input in operation.inputs]
    last_output_node_inputs =  [input for input in operation.inputs]        
    print('###############\n')

