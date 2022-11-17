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
for operation in network[::-1]:
    output_node_gradient = operation.outputs[0].gradient if network.index(operation)<len(network)-1 else [1, 0]
    print('###############\nOperation:', operation.op.__class__.__name__)
    print('Gradient of layer after this:', output_node_gradient)
    for i, gradient in enumerate(operation.op.backward(operation, output_node_gradient)):
        operation.inputs[i].gradient = gradient

        print('\nInput:', operation.inputs[i].name,'->' ,operation.inputs[i].value)
        print('Gradient:', operation.inputs[i].gradient)        
    print('###############\n')
