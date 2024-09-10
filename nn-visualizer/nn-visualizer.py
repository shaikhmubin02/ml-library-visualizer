import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class NeuralNetwork:
    def __init__(self, layer_sizes, activation_functions, use_batch_norm=False, dropout_rate=0.0, l1_reg=0.0, l2_reg=0.0):
        self.layer_sizes = layer_sizes
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.zeros((y, 1)) for y in layer_sizes[1:]]
        self.activation_functions = activation_functions
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.loss_history = []
        self.accuracy_history = []
        self.learning_rates = [0.1] * (len(layer_sizes) - 1)
        
        if use_batch_norm:
            self.batch_norm_params = [{'gamma': np.ones((y, 1)), 'beta': np.zeros((y, 1))} for y in layer_sizes[1:]]

    def forward(self, a, training=False):
        self.layer_outputs = [a]
        self.z_values = []
        
        for i, (w, b, activation) in enumerate(zip(self.weights, self.biases, self.activation_functions)):
            z = np.dot(w, a) + b
            self.z_values.append(z)
            
            if self.use_batch_norm:
                z = self.batch_normalize(z, i, training)
            
            a = self.activate(z, activation)
            
            if training and i < len(self.weights) - 1:
                a = self.apply_dropout(a)
            
            self.layer_outputs.append(a)
        
        return a

    def batch_normalize(self, z, layer, training):
        if training:
            mean = np.mean(z, axis=1, keepdims=True)
            var = np.var(z, axis=1, keepdims=True)
            z_norm = (z - mean) / np.sqrt(var + 1e-8)
            return self.batch_norm_params[layer]['gamma'] * z_norm + self.batch_norm_params[layer]['beta']
        else:
            return z  # For simplicity, we're not implementing running averages for inference

    def apply_dropout(self, a):
        mask = np.random.binomial(1, 1 - self.dropout_rate, size=a.shape) / (1 - self.dropout_rate)
        return a * mask

    def activate(self, z, activation):
        if activation == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-z))
        elif activation == 'relu':
            return np.maximum(0, z)
        elif activation == 'tanh':
            return np.tanh(z)
        elif activation == 'linear':
            return z

    def activate_derivative(self, z, activation):
        if activation == 'sigmoid':
            s = self.activate(z, 'sigmoid')
            return s * (1 - s)
        elif activation == 'relu':
            return (z > 0).astype(float)
        elif activation == 'tanh':
            return 1 - np.tanh(z)**2
        elif activation == 'linear':
            return np.ones_like(z)

    def backprop(self, x, y):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        # Forward pass
        self.forward(x, training=True)

        # Backward pass
        delta = (self.layer_outputs[-1] - y) * self.activate_derivative(self.z_values[-1], self.activation_functions[-1])
        nabla_w[-1] = np.dot(delta, self.layer_outputs[-2].transpose())
        nabla_b[-1] = delta

        for l in range(2, len(self.layer_sizes)):
            delta = np.dot(self.weights[-l+1].transpose(), delta) * self.activate_derivative(self.z_values[-l], self.activation_functions[-l])
            nabla_w[-l] = np.dot(delta, self.layer_outputs[-l-1].transpose())
            nabla_b[-l] = delta

        # Add regularization gradients
        for i in range(len(nabla_w)):
            nabla_w[i] += self.l2_reg * self.weights[i]
            nabla_w[i] += self.l1_reg * np.sign(self.weights[i])

        return nabla_w, nabla_b

    def train(self, X, y, epochs, batch_size, learning_rate_scheduler, task='classification'):
        n_batches = X.shape[1] // batch_size
        for epoch in range(epochs):
            # Shuffle the data
            permutation = np.random.permutation(X.shape[1])
            X_shuffled = X[:, permutation]
            y_shuffled = y[:, permutation]
            
            for batch in range(n_batches):
                start = batch * batch_size
                end = start + batch_size
                X_batch = X_shuffled[:, start:end]
                y_batch = y_shuffled[:, start:end]
                
                nabla_w = [np.zeros(w.shape) for w in self.weights]
                nabla_b = [np.zeros(b.shape) for b in self.biases]
                
                for x, y_true in zip(X_batch.T, y_batch.T):
                    delta_nabla_w, delta_nabla_b = self.backprop(x.reshape(-1, 1), y_true.reshape(-1, 1))
                    nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
                    nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                
                # Update weights and biases
                self.weights = [w - (lr / batch_size) * nw for w, lr, nw in zip(self.weights, self.learning_rates, nabla_w)]
                self.biases = [b - (lr / batch_size) * nb for b, lr, nb in zip(self.biases, self.learning_rates, nabla_b)]
            
            # Calculate and store loss and accuracy
            y_pred = self.predict(X)
            if task == 'classification':
                loss = np.mean(np.sum((y_pred - y)**2, axis=0))
                accuracy = accuracy_score(y.argmax(axis=0), y_pred.argmax(axis=0))
                self.accuracy_history.append(accuracy)
            else:  # regression
                loss = mean_squared_error(y.flatten(), y_pred.flatten())
            self.loss_history.append(loss)
            
            # Update learning rates
            self.learning_rates = learning_rate_scheduler(self.learning_rates, epoch)

    def predict(self, X):
        return np.array([self.forward(x.reshape(-1, 1), training=False).flatten() for x in X.T]).T

def visualize_3d_network(nn):
    edge_x = []
    edge_y = []
    edge_z = []
    node_x = []
    node_y = []
    node_z = []
    node_text = []
    
    for i, layer_size in enumerate(nn.layer_sizes):
        layer_x = [i] * layer_size
        layer_y = list(range(layer_size))
        layer_z = [0] * layer_size
        
        node_x.extend(layer_x)
        node_y.extend(layer_y)
        node_z.extend(layer_z)
        
        for j in range(layer_size):
            node_text.append(f"Layer {i+1}, Neuron {j+1}")
        
        if i < len(nn.layer_sizes) - 1:
            for j in range(layer_size):
                for k in range(nn.layer_sizes[i+1]):
                    edge_x.extend([i, i+1, None])
                    edge_y.extend([j, k, None])
                    edge_z.extend([0, 0, None])

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            color=[],
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_adjacencies = []
    for node, x, y, z in zip(range(len(node_x)), node_x, node_y, node_z):
        node_adjacencies.append(len([nx for nx, ny, nz in zip(node_x, node_y, node_z) if abs(nx-x) <= 1 and abs(ny-y) <= 1 and abs(nz-z) <= 1]))

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='3D Neural Network Visualization',
                        showlegend=False,
                        scene=dict(
                            xaxis=dict(title='Layers'),
                            yaxis=dict(title='Neurons'),
                            zaxis=dict(title=''),
                        ),
                        margin=dict(b=0, l=0, r=0, t=40),
                        hovermode='closest',
                        annotations=[dict(
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002 )]
                        ))
    
    return fig

def visualize_3d_weights(nn):
    data = []
    
    for i, w in enumerate(nn.weights):
        x, y = np.meshgrid(range(w.shape[1]), range(w.shape[0]))
        z = w
        
        trace = go.Surface(
            x=x,
            y=y,
            z=z,
            colorscale='RdBu',
            name=f'Layer {i+1} to {i+2}'
        )
        data.append(trace)
    
    layout = go.Layout(
        title='3D Weight Visualization',
        scene=dict(
            xaxis_title='Input Neuron',
            yaxis_title='Output Neuron',
            zaxis_title='Weight Value'
        ),
        autosize=False,
        width=800,
        height=500,
    )
    
    fig = go.Figure(data=data, layout=layout)
    return fig

def visualize_3d_activations(nn, input_data):
    # Ensure input_data has the correct shape
    if input_data.shape[0] != nn.layer_sizes[0]:
        # If the input doesn't match the network's input layer, use a random input
        input_data = np.random.rand(nn.layer_sizes[0], 1)
    else:
        input_data = input_data.reshape(nn.layer_sizes[0], 1)
    
    activations = nn.forward(input_data, training=False)
    
    data = []
    
    for i, activation in enumerate(nn.layer_outputs):
        x = np.arange(activation.shape[0])
        y = np.full_like(x, i)
        z = activation.flatten()
        
        trace = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=5,
                color=z,
                colorscale='Viridis',
                opacity=0.8
            ),
            name=f'Layer {i+1}'
        )
        data.append(trace)
    
    layout = go.Layout(
        title='3D Activation Visualization',
        scene=dict(
            xaxis_title='Neuron Index',
            yaxis_title='Layer',
            zaxis_title='Activation Value'
        ),
        autosize=False,
        width=800,
        height=500,
    )
    
    fig = go.Figure(data=data, layout=layout)
    return fig

def visualize_gradients(nn, X, y):
    gradients = []
    for x, y_true in zip(X.T, y.T):
        nabla_w, _ = nn.backprop(x.reshape(-1, 1), y_true.reshape(-1, 1))
        gradients.append([np.mean(np.abs(nw)) for nw in nabla_w])
    
    gradients = np.array(gradients).T
    
    fig = go.Figure()
    for i, layer_gradients in enumerate(gradients):
        fig.add_trace(go.Box(y=layer_gradients, name=f'Layer {i+1}'))
    
    fig.update_layout(
        title='Gradient Magnitudes Across Layers',
        yaxis_title='Absolute Gradient',
        boxmode='group'
    )
    
    return fig

def feature_importance(nn, X):
    importance = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        x = X[:, i]
        x_reshaped = x.reshape(nn.layer_sizes[0], -1)
        nn.forward(x_reshaped)
        importance[i] = np.abs(nn.weights[0] @ x_reshaped).sum()
    return importance / X.shape[0]

def learning_rate_scheduler(learning_rates, epoch):
    return [lr * 0.99 for lr in learning_rates]  # Simple decay

def visualize_decision_boundary(nn, X, y):
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = nn.predict(np.c_[xx.ravel(), yy.ravel()].T)
    Z = Z.reshape(xx.shape)

    fig = go.Figure(data=[
        go.Contour(x=xx[0], y=yy[:, 0], z=Z, colorscale='RdBu', opacity=0.5),
        go.Scatter(x=X[0, y.argmax(axis=0)==0], y=X[1, y.argmax(axis=0)==0], mode='markers', marker=dict(color='blue', size=8), name='Class 0'),
        go.Scatter(x=X[0, y.argmax(axis=0)==1], y=X[1, y.argmax(axis=0)==1], mode='markers', marker=dict(color='red', size=8), name='Class 1')
    ])

    fig.update_layout(title='Decision Boundary', xaxis_title='Feature 1', yaxis_title='Feature 2')
    return fig

st.markdown("<h1 style='margin-bottom: 50px;'>Neural Network Visualizer</h1>", unsafe_allow_html=True)

# Sidebar for network configuration
st.sidebar.header('Network Configuration')

github_logo = "https://cdn-icons-png.flaticon.com/128/733/733609.png"
linkedin_logo = "https://cdn-icons-png.flaticon.com/128/2504/2504923.png"

st.sidebar.markdown(
    f'''
    <a href="https://www.linkedin.com/in/shaikhmubin/" target="_blank">
        <img src="{linkedin_logo}" width="22" style="margin-right: 10px">
    </a>
    <a href="https://github.com/shaikhmubin02/ml-library-visualizer" target="_blank">
        <img src="{github_logo}" width="22">
    </a>
    ''',
    unsafe_allow_html=True
)

st.sidebar.markdown('---') 

task = st.sidebar.selectbox('Task', ['Classification', 'Regression'])
num_layers = st.sidebar.slider('Number of layers', 2, 5, 3)
layer_sizes = [st.sidebar.slider(f'Neurons in layer {i+1}', 1, 50, 10) for i in range(num_layers)]
activation_functions = [st.sidebar.selectbox(f'Activation for layer {i+1}', ['sigmoid', 'relu', 'tanh', 'linear']) for i in range(num_layers-1)]
use_batch_norm = st.sidebar.checkbox('Use Batch Normalization', value=False)
dropout_rate = st.sidebar.slider('Dropout Rate', 0.0, 0.5, 0.0, 0.1)
l1_reg = st.sidebar.slider('L1 Regularization', 0.0, 0.1, 0.0, 0.01)
l2_reg = st.sidebar.slider('L2 Regularization', 0.0, 0.1, 0.0, 0.01)

# Create neural network
nn = NeuralNetwork(layer_sizes, activation_functions, use_batch_norm, dropout_rate, l1_reg, l2_reg)

# Training data
st.sidebar.header('Training Data')
n_samples = st.sidebar.slider('Number of samples', 100, 1000, 500)
n_features = layer_sizes[0]
n_classes = layer_sizes[-1]

if task == 'Classification':
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_features, n_redundant=0, n_classes=n_classes, n_clusters_per_class=1)
    y = np.eye(n_classes)[y]  # One-hot encode output
else:  # Regression
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=0.1)
    y = y.reshape(-1, 1)

X = (X - X.min()) / (X.max() - X.min())  # Normalize input
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Training
st.sidebar.header('Training')
epochs = st.sidebar.slider('Epochs', 10, 1000, 100)
batch_size = st.sidebar.slider('Batch Size', 1, 100, 32)

if st.sidebar.button('Train Network'):
    nn.train(X_train, y_train, epochs, batch_size, learning_rate_scheduler, task.lower())
    st.success('Training complete!')

# 3D Visualizations
fig_3d_network = visualize_3d_network(nn)
st.plotly_chart(fig_3d_network)

fig_3d_weights = visualize_3d_weights(nn)
st.plotly_chart(fig_3d_weights)

sample_input = X_train[:, 0].reshape(-1, 1)  # Use the first sample from X_train
fig_3d_activations = visualize_3d_activations(nn, sample_input)
st.plotly_chart(fig_3d_activations)

# Loss and Accuracy Visualization
if nn.loss_history:
    st.header('Training Metrics')
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Training Loss', 'Training Accuracy'))
    fig.add_trace(go.Scatter(y=nn.loss_history, mode='lines', name='Loss'), row=1, col=1)
    if task == 'Classification':
        fig.add_trace(go.Scatter(y=nn.accuracy_history, mode='lines', name='Accuracy'), row=1, col=2)
    fig.update_layout(height=400, width=800)
    st.plotly_chart(fig)

# Gradient Visualization
if nn.loss_history:
    st.header('Gradient Visualization')
    fig_gradients = visualize_gradients(nn, X_train, y_train)
    st.plotly_chart(fig_gradients)

# Feature Importance
importance = feature_importance(nn, X_train)
fig_importance = go.Figure(data=go.Bar(x=list(range(n_features)), y=importance))
fig_importance.update_layout(title='Feature Importance', xaxis_title='Feature Index', yaxis_title='Importance')
st.plotly_chart(fig_importance)

# Decision Boundary Visualization (for 2D classification problems)
if task == 'Classification' and n_features == 2:
    st.header('Decision Boundary')
    fig_decision_boundary = visualize_decision_boundary(nn, X, y)
    st.plotly_chart(fig_decision_boundary)

# Model Summary
st.header('Model Summary')
total_params = sum(w.size + b.size for w, b in zip(nn.weights, nn.biases))
st.write(f"Total parameters: {total_params}")
for i, (w, b) in enumerate(zip(nn.weights, nn.biases)):
    st.write(f"Layer {i+1}: Weights shape {w.shape}, Biases shape {b.shape}")

# Prediction
st.header('Make Prediction')
st.write('Enter values for each input neuron:')
input_values = [st.slider(f'Input {i+1}', 0.0, 1.0, 0.5) for i in range(nn.layer_sizes[0])]
input_data = np.array(input_values).reshape(-1, 1)

if st.button('Predict'):
    prediction = nn.forward(input_data, training=False)
    st.write('Prediction:', prediction.flatten())
    fig_prediction = visualize_3d_activations(nn, input_data)
    st.plotly_chart(fig_prediction)

# Evaluation
if nn.loss_history:
    st.header('Model Evaluation')
    y_pred = nn.predict(X_test)
    if task == 'Classification':
        accuracy = accuracy_score(y_test.argmax(axis=0), y_pred.argmax(axis=0))
        st.write(f'Test Accuracy: {accuracy:.2f}')
    else:  # Regression
        mse = mean_squared_error(y_test.flatten(), y_pred.flatten())
        st.write(f'Test Mean Squared Error: {mse:.4f}') 


# ... (previous code remains unchanged)

# Add this section at the end of the file, just before the last closing quotation mark of the markdown explanation

st.markdown("""
## How Neural Networks Work

1. **Initialization**: The network is initialized with random weights and biases.
2. **Forward Propagation**: Input data is passed through the network, layer by layer.
3. **Loss Calculation**: The difference between the network's output and the true values is calculated.
4. **Backpropagation**: The error is propagated backwards through the network to calculate gradients.
5. **Weight Update**: The weights and biases are updated based on the calculated gradients.
6. **Repeat**: Steps 2-5 are repeated for multiple epochs to improve the network's performance.

This visualizer allows you to see various aspects of neural networks, from their structure to their training process and decision boundaries.

### Features

1. **3D Network Visualization**: See the structure of your neural network in 3D.
2. **3D Weight Visualization**: Visualize the weights between layers in 3D.
3. **3D Activation Visualization**: See how activations propagate through the network for a given input.
4. **Training Metrics**: Monitor loss and accuracy during training.
5. **Gradient Visualization**: Understand how gradients change across layers during training.
6. **Feature Importance**: See which input features have the most impact on the network's decisions.
7. **Decision Boundary Visualization**: For 2D classification problems, visualize how the network separates classes.
8. **Model Summary**: Get an overview of the network's architecture and total parameters.
9. **Interactive Prediction**: Input your own values and see the network's prediction.
10. **Customizable Architecture**: Adjust the number of layers, neurons, and choose activation functions.
11. **Regularization Options**: Experiment with dropout, L1, and L2 regularization.
12. **Batch Normalization**: Option to use batch normalization in your network.
13. **Classification and Regression**: Support for both classification and regression tasks.

These tools can help in understanding neural networks better, from their basic structure to their training dynamics and decision-making process.
""")