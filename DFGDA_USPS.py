import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from collections import deque
import numpy as np
from torch.optim import Optimizer
import random
import math


# Set fixed random seed for reproducibility
torch.manual_seed(42)

# A CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 max pooling
        self.fc1 = nn.Linear(64 * 3 * 3, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, 10)  # 10 classes for digits 0-9

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # Conv1 + ReLU + MaxPool
        x = self.pool(torch.relu(self.conv2(x)))  # Conv2
        x = self.pool(torch.relu(self.conv3(x)))  # Conv3
        x = x.view(-1, 64 * 3 * 3)  # Flatten
        x = torch.relu(self.fc1(x))  # Fully connected layer 1
        x = self.fc2(x)  # Fully connected layer 2
        return x

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load USPS dataset
train_dataset = torchvision.datasets.USPS(root='./data', train=True, download=True, transform=transform)

# Split dataset into training and validation
num_train = int(len(train_dataset) * 0.8)
num_valid = len(train_dataset) - num_train
train_data, valid_data = torch.utils.data.random_split(train_dataset, [num_train, num_valid])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_data, batch_size=64, shuffle=False, num_workers=4)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#DF-GDA Optimizer
class GDAOptimizer_frac:
    def __init__(self, params, num_states= 256, initial_temp=1.0, initial_rate=0.992, f_min=1.0, f_max=1.0, window_size=10):
        self.params = list(params)
        self.temperature = initial_temp
        self.initial_rate = initial_rate
        self.num_states = num_states
        self.f_min = f_min
        self.f_max = f_max
        self.window_size = window_size
        self.loss_history = deque(maxlen=window_size)
        self.mean_field = {param: torch.zeros_like(param) for param in self.params}
        self.ft_t = 0.0
        self.epoch = 0

        #f_min can be selected to adjust the fraction of parameters. 
        # For example, f_min = 0.25 means only 0.25 of the parameters are updated. 
        # As for default, it is set to 1.0 for a fair comparison between DF-GDA and SGD as SGD also update all the parameters.

    def soft_quantization(self, param):
        state_values = torch.linspace(param.min(), param.max(), self.num_states).to(param.device)
        distances = torch.abs(param.unsqueeze(-1) - state_values)
        softmax_probabilities = F.softmax(-distances / self.temperature, dim=-1)
        return torch.sum(state_values * softmax_probabilities, dim=-1)
    
    def calculate_entropy(self):
        total_entropy = 0
        for param in self.params:
            if param.grad is not None:
                p = F.softmax(param.data.flatten() / self.temperature, dim=0)
                entropy = -torch.sum(p * torch.log(p + 1e-10))  # Adding a small number to avoid log(0)
                total_entropy += entropy
        return total_entropy

    def update_fraction(self):
        if len(self.loss_history) < self.window_size:
            return self.f_max  # Use the maximum fraction if there's not enough history

        # Calculate the average change in loss
        loss_changes = [abs(self.loss_history[i] - self.loss_history[i - 1]) for i in range(1, len(self.loss_history))]
        avg_loss_change = np.mean(loss_changes)

        # Calculate the maximum observed average change in loss
        max_loss_change = max(loss_changes) if loss_changes else 1.0  # Avoid division by zero

        # Update fraction inversely proportional to the average loss change
        f_t = self.f_max - (avg_loss_change / max_loss_change) * (self.f_max - self.f_min)
        
        return max(self.f_min, min(f_t, self.f_max))  # Ensure f_t is within [f_min, f_max]

    def step(self, loss, model, data, target, loss_fn):
        loss.backward()

        with torch.no_grad():
            total_entropy = self.calculate_entropy()

            # Dynamically calculate the update fraction
            self.ft_t = self.update_fraction()
            num_params_to_update = int(len(self.params) * self.ft_t)
            params_to_update = torch.randperm(len(self.params))[:num_params_to_update]

            for idx in params_to_update:
                param = self.params[idx]
                if param.grad is not None:
                    # Mean Field Gradient Update
                    self.mean_field[param] = 0.9 * self.mean_field[param]  + 0.1 *param.grad

                    noise = torch.randn_like(param.data) * self.temperature
                    candidate_param = param.data - 0.01 * self.mean_field[param] + noise
                    quantized_param = self.soft_quantization(candidate_param)

                    original_param = param.data.clone()
                    param.data.copy_(quantized_param)

                    model.zero_grad()
                    torch.set_grad_enabled(True)
                    output = model(data)
                    new_loss = loss_fn(output, target)
                    new_loss.backward(retain_graph=True)

                    energy_diff = new_loss - loss
                    acceptance_prob = torch.exp(-energy_diff / self.temperature)
                    
                    num_states = 1024

                    #GDA acceptance criteria
                    generation_prob =  1 / num_states #generation function

                    if (energy_diff > generation_prob  * acceptance_prob):
                        param.data.copy_(original_param)

                    torch.set_grad_enabled(False)

            self.temperature *= self.initial_rate ** (1 + total_entropy / len(self.params))


    def zero_grad(self):
        for param in self.params:
            param.grad = None

    def update_loss_history(self, loss_value):
        self.loss_history.append(loss_value)
        
    def increment_epoch(self):
        self.epoch += 1

      
# Simulated Annealing Optimizer:         
class SimulatedAnnealing(Optimizer):
    def __init__(self, params, initial_temp=1.0, cooling_rate=0.9, min_temp=1e-3, sampler=None):
        defaults = {'initial_temp': initial_temp, 'cooling_rate': cooling_rate, 'min_temp': min_temp}
        super(SimulatedAnnealing, self).__init__(params, defaults)
        self.temperature = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.sampler = sampler  # Sampler for generating random noise for parameter updates
    
    def step(self, closure=None):
        """Performs a single optimization step based on simulated annealing.
        
        Args:
            closure (callable, optional): A closure that re-evaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # Update each parameter using Simulated Annealing
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                
                # Get current parameter state and gradient
                current_param_data = param.data
                current_grad = param.grad.data

                # Propose a new candidate state using random noise (from the sampler)
                noise = self.sampler.sample(current_param_data.size())
                candidate_param = current_param_data - 0.01 * current_grad + noise

                # Calculate the loss for the proposed candidate
                original_loss = loss.item()
                param.data = candidate_param  # Temporarily set the parameter to candidate
                new_loss = closure().item()  # Evaluate new loss

                # Compute the energy difference (loss difference)
                energy_diff = new_loss - original_loss

                # Convert energy_diff to a tensor before applying torch.exp()
                energy_diff_tensor = torch.tensor(energy_diff, device=param.device)

                # Acceptance probability based on energy difference and temperature
                if energy_diff_tensor <= 0 or random.random() < torch.exp(-energy_diff_tensor / self.temperature).item():
                    # Accept the new state (already set)
                    pass
                else:
                    # Reject the new state, revert to the original state
                    param.data = current_param_data

        # Decrease the temperature
        self.temperature = max(self.min_temp, self.temperature * self.cooling_rate)

        return loss
        
        
class UniformSampler:
    def __init__(self, minval, maxval, cuda=False):
        self.minval = minval
        self.maxval = maxval
        self.cuda = cuda

    def sample(self, size):
        if self.cuda:
            return (self.minval - self.maxval) * torch.rand(size).cuda() + self.maxval
        else:
            return (self.minval - self.maxval) * torch.rand(size) + self.maxval



# Dataset and DataLoader setup


#Training using DF-GDA
def train_GDA(model, device, train_loader, valid_loader, optimizer, loss_fn, epochs):

    model.to(device)
    for epoch in range(epochs):

        model.train()
        total_train_loss = 0
        total_entropy = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            optimizer.update_loss_history(loss.item())
            optimizer.step(loss, model, data, target, loss_fn)
            total_train_loss += loss.item()
            total_entropy += optimizer.calculate_entropy().item()

        optimizer.increment_epoch()
        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = loss_fn(output, target)
                total_valid_loss += loss.item()
                
        print(f'Epoch: {epoch + 1} Training Loss: {total_train_loss / len(train_loader):.6f} Validation Loss: {total_valid_loss / len(valid_loader):.6f}')


#Training using SGD
def train_SGD(model, device, train_loader, valid_loader, optimizer, loss_fn, epochs):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            for data, target in valid_loader:

                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = loss_fn(output, target)
                total_valid_loss += loss.item()

        print(f'Epoch: {epoch + 1}, Training Loss: {total_train_loss / len(train_loader):.6f}, Validation Loss: {total_valid_loss / len(valid_loader):.6f}')


loss_fn = nn.CrossEntropyLoss()



sampler = UniformSampler(minval=-0.5, maxval=0.5, cuda=torch.cuda)


#CNN model for DF-GDA
model_GDA = CNN().to(device)

#CNN model for SGD
model_SGD = CNN().to(device)

#CNN model for SA
model_SA= CNN().to(device)


sampler = UniformSampler(minval=-0.5, maxval=0.5, cuda=torch.cuda)

optimizer_SGD = torch.optim.SGD(model_SGD.parameters(), lr=0.001)
optimizer_GDA = GDAOptimizer_frac(model_GDA.parameters(), initial_temp=1.0, initial_rate=0.992)
optimizer_SA = SimulatedAnnealing(model_SA.parameters(), sampler=sampler)


print('Start training using DF-GDA')
train_GDA(model_GDA, device, train_loader, valid_loader, optimizer_GDA, loss_fn, epochs=10)

print('Start training using SGD')
train_SGD(model_SGD, device, train_loader, valid_loader, optimizer_SGD, loss_fn, epochs=10)
