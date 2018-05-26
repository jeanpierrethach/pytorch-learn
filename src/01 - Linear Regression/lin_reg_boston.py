from sklearn.datasets import load_boston

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# HYperparameters
input_size = 13
output_size = 1
num_epochs = 1000
learning_rate = 1e-6

boston = load_boston()

features = torch.from_numpy(np.array(boston.data)).float()
labels = torch.from_numpy(np.array(boston.target)).float().view(-1,1)

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out

model = LinearRegression()

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    targets = labels

    # Forward pass
    outputs = model(features)
    loss = criterion(outputs, targets)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 20 == 0:
        print('Epoch[{}/{}], loss: {:.6f}'.format(epoch+1, num_epochs, loss.item()))

model.eval()

predicted = model(features).detach().numpy()
labels = np.array(boston.target)

fig, ax = plt.subplots()
ax.scatter(labels, predicted)
ax.plot([labels.min(), labels.max()], [labels.min(), labels.max()], 'k--', lw=3)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')