import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

# This is the parameter we want to optimize -> requires_grad=True
w = torch.tensor(1.0, requires_grad=True)

steps = 1000
lr = 0.01

for _ in range(steps):
    # forward pass to compute loss
    y_predicted = w * x
    loss = (y_predicted - y)**2
    print('loss:', loss)

    # backward pass to compute gradient dLoss/dw
    loss.backward()
    print('grad:', w.grad)

    # update weights
    # next forward and backward pass...

    # continue optimizing:
    # update weights, this operation should not be part of the computational graph
    with torch.no_grad():
        w -= lr * w.grad
        print('w:', w)
    # don't forget to zero the gradients
    w.grad.zero_()

    # next forward and backward pass...
