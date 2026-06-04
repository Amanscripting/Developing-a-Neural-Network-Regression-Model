## Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
The objective of this experiment is to design, implement, and evaluate a Deep Learning–based Neural Network regression model to predict a continuous output variable from a given set of input features. The task is to preprocess the data, construct a neural network regression architecture, train the model using backpropagation and gradient descent, and evaluate its performance using appropriate regression metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² score.

## Neural Network Model
Include the neural network model diagram.
<img width="1032" height="668" alt="image" src="https://github.com/user-attachments/assets/a4beab91-0fa0-4e85-96d1-86cb35e611ea" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:Aman Alam

### Register Number:212224240011

```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,8)
        self.fc2=nn.Linear(8,10)
        self.fc3=nn.Linear(10,1)
        self.relu=nn.ReLU()
        self.history={'loss':[]}
    def forward(self,x):
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        return x



# Initialize the Model, Loss Function, and Optimizer
ai_brain=NeuralNet()
criterion=nn.MSELoss()
optimizer=optim.RMSprop(ai_brain.parameters(),lr=0.001)




def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
      optimizer.zero_grad()
      Loss=criterion(ai_brain(X_train),y_train)
      Loss.backward()
      optimizer.step()
      ai_brain.history['loss'].append(Loss.item())
      if epoch % 200 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {Loss.item():.6f}')


```

### Dataset Information
Include screenshot of the generated data
<img width="190" height="174" alt="image" src="https://github.com/user-attachments/assets/82b2eab2-09f2-4cdf-8250-75a761e09a93" />


### OUTPUT
<img width="443" height="323" alt="image" src="https://github.com/user-attachments/assets/3d809dde-021c-4143-a06e-50685ca17eba" />


### Training Loss Vs Iteration Plot

<img width="897" height="694" alt="image" src="https://github.com/user-attachments/assets/77befb95-3a0f-418b-8fa8-337cabc9d833" />


### New Sample Data Prediction

<img width="380" height="21" alt="image" src="https://github.com/user-attachments/assets/0c35b953-3ade-488a-bcaf-c7da7fcb6a9f" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
