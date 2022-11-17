from nodes import TensorNode

e = 2.718281828459045


class Op:
    '''Defines a particular operation. To be used in an particular Op(eration)Node'''

    def forward(self, inputs):
        # Given the inputs, compute the outputs
        pass

    def backward(self, parent, outputs_gradient):
        # Given the gradient of the loss with respect ot the outputs
        # compute the gradient of the loss wrt to the inputs
        # does not compute local derivative but computes the accumulated gradient over the loss 
        # So given the gradient for the output node B it computes the gradient for A
        pass

class Plus(Op):
    '''Summation operation'''

    def forward(self, a,b):
        # a and b are matrices of the same sizes
        return a+b
    
    def backward(self, parent, outputs_gradient):
        # given the gradient for s (output), what should we return as gradient for a and b?

        # SOLUTION
        # 1. Write out the scalar derivative for a single element

        # 2. Use the multivariate chain rule to sum over the elements of the gradient for the output (sum over all outputs)

        # 3. Vectorize the result

        # EXAMPLE
        # gradient for a
        # 1. start with scalar derivative = A∇ᵢⱼ = dl / dAᵢⱼ (so delta loss wrrt delta Aᵢⱼ)
        # Break up with chain rule: dl / dS * dS / DA
        # 2. With the multivariate chain rule sum over all k and l for S: ∑ₖₗ (dl / dSₖₗ) * (dSₖₗ / dAᵢⱼ)
        # (dl / dSₖₗ) is actually givn via the outputs_gradient going backward: S∇ₖₗ
        #  ∑ₖₗ S∇ₖₗ * (dSₖₗ / dAᵢⱼ)
        # (dSₖₗ / dAᵢⱼ) can be worked out by filling in S which is the kl element A+B: (dAₖₗ + Bₖₗ / dAᵢⱼ)
        # ∑ₖₗ S∇ₖₗ * (dAₖₗ + Bₖₗ / dAᵢⱼ)
        # Bₖₗ disappears since there will never be any change in relation to dAᵢⱼ
        # ∑ₖₗ S∇ₖₗ * (dAₖₗ / dAᵢⱼ)
        # dAₖ is only non-zero when i=k and j=l, otherwise it is zero
        # So we remove the sum (since it will all be summing zero elements), except for the case i=k, j=l
        # S∇ᵢⱼ * (dAᵢⱼ / dAᵢⱼ)
        # (dAᵢⱼ / dAᵢⱼ) is always 1, so we end up with only S∇ᵢⱼ
        # A∇ᵢⱼ = S∇ᵢⱼ
        # By symmetrix we know that the same reuslt should hold for B∇ᵢⱼ
        # 3. Vectorization is easy because ij is always the same so A∇ = S∇
        # So we can take the gradient for S that the system gives as and return it as the gradient for A and B
        return outputs_gradient, outputs_gradient

class Sigmoid(Op):
    '''Sigmoid operation'''
    def forward(self, x):
        # x is a tensor of any shape
        sigx = [1 / (1 + e**-i) for i in x.value]
        # Not sure why the lessons save the context in this way
        # context['sigx'] = sigx
        return TensorNode(sigx)

    def backward(self, parent, outputs_gradient):
        # EXAMPLE
        # gradient for X is X∇ = dl / dX∇
        # Break up with chain rule: X∇ = (dl / dY) * (dY / dX)
        # Fill in (dl / dY) with Y∇ since this is given via the outputs_gradients via the backward
        # X∇ = Y∇ * (dY / dX)
        # Start with single scalar derivative (so sum over the elements of the outputs): 
        # X∇ᵢⱼₖ = ∑ₐ₆꜀ Y∇ₐ₆꜀ * (dYₐ₆꜀ / dXᵢⱼₖ)
        # Fill in dYₐ₆꜀ with σ(Xₐ₆꜀) since this is a forward pass calculation
        # X∇ᵢⱼₖ = ∑ₐ₆꜀ Y∇ₐ₆꜀ * (dσ(Xₐ₆꜀) / dXᵢⱼₖ)
        # when ijk≠abc then derivative is zero since dσ(Xₐ₆꜀) / dXᵢⱼₖ change will be zero
        # So all elements will sum to zero except when ijk = abc, this is the only part in the sum with a value
        # Y∇ᵢⱼₖ * (dσ(Xᵢⱼₖ) / dXᵢⱼₖ)
        # Now notice how (dσ(Xᵢⱼₖ) / dXᵢⱼₖ) is simply the scalar derivative of the sigmoid function, we know the sigmoid function: dσ = σ(1-σ), so we fill that in
        # Y∇ᵢⱼₖ * σ(Xᵢⱼₖ) (1 - σ(Xᵢⱼₖ))
        # Now we already know the output of the σ in the computation graph to be Yᵢⱼₖ, so we fill that in
        # Y∇ᵢⱼₖ * Yᵢⱼₖ (1 - Yᵢⱼₖ)
        # This is our scalar derivative so we know vectorize this to
        # X∇ = Y∇ ⨷ Y ⨷ (1 - Y)     (⨷ = element-wise multiplication)
        # Now these Ys show that we need more than only outputs-gradients also the forward pass output > this is what the context object is for
        # print(parent.outputs[0].gradient)
        return [[parent.outputs[0].gradient[idx] * i * (1- i) for idx, i in enumerate(j.value)]  for j in parent.outputs]
        # Single case: [outputs_gradient * parent.outputs[0].value * (1- parent.outputs[0].value)]

class SoftMax(Op):
    '''Apply softmax to the layer'''
    def forward(self, s):
        sum_inputs = 0
        for i in s.value: sum_inputs+= e**i
        return TensorNode(init_value=[e**i/sum_inputs for i in s.value])
    
    def backward(self, parent, outputs_gradient):
        gradient = []
        for idx, value in enumerate(parent.outputs[0].value):
            if outputs_gradient[idx] == 1:
                gradient.append(-1 * (1-value))
            else:
                gradient.append(-value * -1)
        return [gradient]
        

class RowSum(Op):
    '''Sum along elements per row, operation. Matrix X becomes vector y'''

    def forward(self, x):
        # x is a matrix
        sumd = x.sum(axis=1)

        return sumd

    def backward(self, parent, outputs_gradient):
        # EXAMPLE
        # gradient for X is X∇ = dl / dX∇
        # Break up with chain rule: X∇ = (dl / dy) * (dy / dX)
        # Fill in (dl / dy) with y∇ since this is given via the outputs_gradients via the backward
        # X∇ = y∇ * (dy / dX)
        # Start with single scalar derivative (so sum over the elements of the outputs): 
        # X∇ᵢⱼ = ∑ₖ y∇ₖ * (dyₖ / dXᵢⱼ)
        # Now we know what results from the forward pass, namely  ∑ₗ Xₖₗ (summing the rows), so we substitute yₖ with that
        # X∇ᵢⱼ = ∑ₖ y∇ₖ * (d ∑ₗ Xₖₗ / dXᵢⱼ)
        # We work this sum out in fornt, so we get a sum k over l
        # ∑ₖₗ y∇ₖ * (dXₖₗ / dXᵢⱼ)
        # Now only when ij=kl will the derivative be non-zero, so all terms of this sum are zero and we keep only the single remaining non-zero term
        # y∇ᵢ * (dXᵢⱼ / dXᵢⱼ)
        # We know that (dXᵢⱼ / dXᵢⱼ) is 1, so:
        # y∇ᵢ is our result.
        # Now we vectorize: y∇ 1ᵀ
        n, m = outputs_gradient.shape[0], parent.outputs[0].value
        return outputs_gradient[:, None].expand(n,m)

class Expand(Op):
    '''Expands a scalar to a matrix of a given size'''

    def forward(self, x, size):
        # x is a scalar
        return np.full(x, size=size)

    def backward(self, parent, outputs_gradient):
        return outputs_gradient.sum(), None # No gradient returned for 'size'

class Multiply(Op):
    '''Multiplies input tensor by weights tensor'''

    def forward(self, a, b):
        # x is a scalar
        return a * b

    def backward(self, parent, outputs_gradient):
        # ASK if this is correct
        # IDEA
        # gradient for a
        # 1. start with scalar derivative = A∇ᵢⱼ = dl / dAᵢⱼ (so delta loss wrrt delta Aᵢⱼ)
        # Break up with chain rule: dl / dS * dS / DA
        # 2. With the multivariate chain rule sum over all k and l for S: ∑ₖₗ (dl / dSₖₗ) * (dSₖₗ / dAᵢⱼ)
        # (dl / dSₖₗ) is actually givn via the outputs_gradient going backward: S∇ₖₗ
        #  ∑ₖₗ S∇ₖₗ * (dSₖₗ / dAᵢⱼ)
        # (dSₖₗ / dAᵢⱼ) can be worked out by filling in S which is the kl element A*B: (dAₖₗ * Bₖₗ / dAᵢⱼ)
        # ∑ₖₗ S∇ₖₗ * (dAₖₗ * Bₖₗ / dAᵢⱼ)
     
        # dAₖ is only non-zero when i=k and j=l, otherwise it is zero
        # So we remove the sum (since it will all be summing zero elements), except for the case i=k, j=l
        # S∇ᵢⱼ * (dAᵢⱼ * Bᵢⱼ / dAᵢⱼ)
        # (dAᵢⱼ * Bᵢⱼ / dAᵢⱼ) is always Bᵢⱼ, so we end up with only S∇ᵢⱼ * Bᵢⱼ
        # A∇ᵢⱼ = S∇ᵢⱼ * Bᵢⱼ
        # By symmetrix we know that the same reuslt should hold for A∇ᵢⱼ
        # 3. Vectorization is easy because ij is always the same so A∇ = S∇B∇
        # So we can take the gradient for S that the system gives as and return it as the gradient for A and B
        num_outputs = int(len(parent.inputs[1].value) / len(parent.inputs[0].value))
        temp = [0] * len(parent.inputs[1].value)
        for i, weight in enumerate(parent.inputs[1].value):
            origin_idx = int(i / num_outputs)
            temp[i] = parent.inputs[0].value[origin_idx] * outputs_gradient[0] # 1 because we cannot change the calculated input, only the added bias

        temp_2 = [0] * len(parent.inputs[0].value)
        for i, weight in enumerate(parent.inputs[1].value):
            target_idx = i % len(parent.inputs[0].value)
            origin_idx = int(i / num_outputs)
            temp_2[target_idx] += weight * parent.inputs[0].value[origin_idx]

        return temp_2, temp # We cannot change input so that will be None
        # Original: outputs_gradient * i for i in parent.inputs[1].value