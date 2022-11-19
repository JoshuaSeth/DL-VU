class OpNode:
    '''Instance of particular operation being applied in the computation '''
    def __init__(self, op, inputs, outputs, name = '') -> None: 
        # op: <Op> the particular operation being used (e.g. multiplicatio or addition)
        self.op = op
        # inputs: List<TensorNode> inputs to the op
        self.inputs = inputs
        # outputs: List<TensorNode>  from the op
        self.outputs = outputs
        # Optional name for debugging
        self.name = name



class TensorNode:
    '''Node that holds a tensor value reuslting from operation or when it is input'''

    def __init__(self, init_value=0, source=None, name='') -> None:
        # value: <tensor>
        self.value = init_value
        # gradient: <tensor> (to be filled in by backprop algorithm)
        self.gradient = None
        # source: <OpNode> pointer to operation node that produced (unless this is an input, then: null)
        self.source = source
        # Optional name for debugging
        self.name = name
    
    def __add__(self, other):
        return TensorNode([i+j for i,j in zip(self.value, other.value)] if len(other.value)== len(self.value) else [other.value[0] + i for i in self.value])

    def __mul__(self, other):
        num_outputs = int(len(other.value) / len(self.value))
        temp = [0] * num_outputs
        for i, weight in enumerate(other.value):
            target_idx = i % num_outputs
            origin_idx = int(i / num_outputs)
            # print(target_idx, origin_idx)
            temp[target_idx] += weight * self.value[origin_idx]
        return TensorNode(temp)
