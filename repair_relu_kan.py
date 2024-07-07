import numpy as np
import torch
import torch.nn as nn


class ReLUKANLayer(nn.Module):
    def __init__(self, input_size: int, g: int, k: int, output_size: int, train_ab: bool = True):
        super().__init__()
        self.g, self.k, self.r = g, k, 4*g*g / ((k+1)*(k+1))
        self.input_size, self.output_size = input_size, output_size
        phase_low = np.arange(-k, g) / g
        phase_height = phase_low + (k+1) / g
        self.phase_low = nn.Parameter(torch.Tensor(np.array([phase_low for i in range(input_size)])),
                                      requires_grad=train_ab)
        self.phase_height = nn.Parameter(torch.Tensor(np.array([phase_height for i in range(input_size)])),
                                         requires_grad=train_ab)
        self.equal_size_conv = nn.Conv2d(1, output_size, (g+k, input_size))

    def forward(self, x):
        # Expand dimensions of x to match the shape of self.phase_low
        x_expanded = x.unsqueeze(2).expand(-1, -1, self.phase_low.size(1))
        
        # Perform the subtraction with broadcasting
        x1 = torch.relu(x_expanded - self.phase_low)
        x2 = torch.relu(self.phase_height - x_expanded)
        
        # Continue with the rest of the operations
        x = x1 * x2 * self.r
        x = x * x
        x = x.reshape((len(x), 1, self.g + self.k, self.input_size))
        x = self.equal_size_conv(x)
        #x = x.reshape((len(x), self.output_size, 1)) 
        x = x.reshape((len(x), self.output_size)) 
        return x

    # def forward(self, x):
    #     x1 = torch.relu(x - self.phase_low)
    #     x2 = torch.relu(self.phase_height - x)
    #     x = x1 * x2 * self.r
    #     x = x * x
    #     x = x.reshape((len(x), 1, self.g + self.k, self.input_size))
    #     x = self.equal_size_conv(x)
    #     x = x.reshape((len(x), self.output_size, 1))
    #     return x


class ReLUKAN(nn.Module):
    def __init__(self, width, grid, k, use_attention=False, use_kan=True):
        super().__init__()
        self.use_attention=use_attention
        self.use_kan=use_kan
        self.width = width
        self.grid = grid
        self.k = k
        self.layers = []

        for i in range(len(width) - 1):
            self.layers.append(ReLUKANLayer(width[i], grid, k, width[i+1]))
            # if len(width) - i > 2:
            #     self.layers.append()
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for rk_layer in self.layers:
            x = rk_layer(x)
        # x = x.reshape((len(x), self.width[-1]))
        return x


def show_base(phase_num, step):
    rk = ReLUKANLayer(1, phase_num, step, 1)
    x = torch.Tensor([np.arange(-600, 1024+600) / 1024]).T
    x1 = torch.relu(x - rk.phase_low)
    x2 = torch.relu(rk.phase_height - x)
    y = x1 * x1 * x2 * x2 * rk.r * rk.r
    for i in range(phase_num+step):
        plt.plot(x, y[:, i:i+1].detach(), color='black')
    plt.show()
    print('1')



if __name__ == '__main__':
    # Define the ReLUKAN with 2 layers: input layer with 10 units, middle layer with 8 units, and output layer with 5 units
    
    
    #self = skan
    # Generate random 10-dimensional input data
    x = torch.Tensor(np.random.rand(1024, 3))

    # Extract x1, x2, x3 from the tensor
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]

    # Generate random values e1 and e2 between 0 and 1
    e1 = torch.Tensor(np.random.rand(1024))
    e2 = torch.Tensor(np.random.rand(1024))

    # Calculate y based on the given formula
    y1 = torch.sin(x1) + torch.cos(x2) + torch.log(x3) + e1
    y2 = x1 + torch.sin(x2) + e2

    # Combine y1 and y2 into a single tensor y with shape (1024, 2)
    y = torch.stack((y1, y2), dim=1)

    rkan1 = ReLUKANLayer(input_size = 3, g=5, k=3, output_size = 3)
    a=rkan1(x); a.shape
    rkan2 = ReLUKANLayer(input_size = 3, g=5, k=3, output_size = 2)
    b = rkan2(a); b.shape

    relu_kan = None
    relu_kan = ReLUKAN([3, 3, 2], 5, 3)

    if torch.cuda.is_available():
        relu_kan = relu_kan.cuda()
        x = x.cuda()
        y = y.cuda()
    elif torch.backends.mps.is_available():
        relu_kan = relu_kan.to("mps")
        x = x.to("mps")
        y = y.to("mps")

    opt = torch.optim.Adam(relu_kan.parameters())
    mse = torch.nn.MSELoss()

    plt.ion()
    losses = []
    for e in range(5000):
        opt.zero_grad()
        pred = relu_kan(x)

        loss = mse(pred, y)
        loss.backward()
        opt.step()
        if e%100==0:
            print(f'Epoch {e}: Loss = {loss.item()}')
            pred = pred.detach()
            plt.clf()
            plt.plot(y.cpu().numpy().flatten(), pred.cpu().numpy().flatten(), 'o')
            plt.pause(0.01)
            print(f'Epoch {e}: Loss = {loss.item()}')
