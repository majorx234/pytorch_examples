# Info
- learn tytorch

# Examples
## 01 - simple feed forward network
- Feedforward01 is a class to abstract the network
  - inherit `nn.Module`
  - constructor
  - forward() -function
  - usage:
    - `my_network = FeedForward01(4,10,2)`
    - `out = my_network.forward(torch.tensor([1.0,2.0,3.0,4.0]))`
    - `print(out)`
