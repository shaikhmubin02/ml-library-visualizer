import streamlit as st
import numpy as np
import pandas as pd
# Remove the following line if you're not using matplotlib directly
# import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
import io
import base64
import time

# Set random seed for reproducibility
np.random.seed(42)

@st.cache_data
def generate_random_data(n_samples, n_features=2):
    """Generate random data points."""
    return np.random.rand(n_samples, n_features)

@st.cache_data
def run_kmeans(data, n_clusters, max_iter=10):
    """Run K-Means clustering algorithm."""
    kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, n_init=1, random_state=42)
    labels = kmeans.fit_predict(data)
    centroids = kmeans.cluster_centers_
    return labels, centroids

def plot_clusters(data, labels, centroids):
    """Plot the clustered data points and centroids in 2D or 3D."""
    color_palette = px.colors.sequential.Rainbow
    if data.shape[1] == 2:
        fig = px.scatter(x=data[:, 0], y=data[:, 1], color=labels, 
                         color_continuous_scale=color_palette,
                         labels={'x': 'Feature 1', 'y': 'Feature 2'},
                         title='K-Means Clustering')
        fig.add_scatter(x=centroids[:, 0], y=centroids[:, 1], mode='markers', 
                        marker=dict(color='black', size=12, symbol='star'),
                        name='Centroids')
    elif data.shape[1] == 3:
        fig = px.scatter_3d(x=data[:, 0], y=data[:, 1], z=data[:, 2], color=labels,
                            color_continuous_scale=color_palette,
                            labels={'x': 'Feature 1', 'y': 'Feature 2', 'z': 'Feature 3'},
                            title='K-Means Clustering (3D)')
        fig.add_scatter3d(x=centroids[:, 0], y=centroids[:, 1], z=centroids[:, 2], mode='markers',
                          marker=dict(color='black', size=7, symbol='diamond'),
                          name='Centroids')
    else:
        raise ValueError("Data must be 2D or 3D for visualization")
    return fig

def elbow_method(data, max_k):
    """Perform elbow method to find optimal K."""
    inertias = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    return inertias

def plot_elbow(inertias):
    """Plot the elbow curve."""
    k_values = list(range(1, len(inertias) + 1))
    fig = px.line(x=k_values, y=inertias, 
                  labels={'x': 'Number of Clusters (K)', 'y': 'Inertia'},
                  title='Elbow Method for Optimal K')
    fig.add_scatter(x=k_values, y=inertias, mode='markers')
    return fig

def plot_silhouette(data, max_k):
    """Plot silhouette scores for different K values."""
    silhouette_scores = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        silhouette_scores.append(score)
    
    k_values = list(range(2, max_k + 1))
    fig = px.line(x=k_values, y=silhouette_scores, 
                  labels={'x': 'Number of Clusters (K)', 'y': 'Silhouette Score'},
                  title='Silhouette Score for Different K Values')
    fig.add_scatter(x=k_values, y=silhouette_scores, mode='markers')
    return fig

def run_kmeans_step(data, n_clusters, current_centroids=None):
    """Run a single step of K-Means clustering."""
    if current_centroids is None:
        # Initialize centroids randomly
        kmeans = KMeans(n_clusters=n_clusters, init='random', n_init=1, max_iter=1, random_state=42)
    else:
        # Use the current centroids
        kmeans = KMeans(n_clusters=n_clusters, init=current_centroids, n_init=1, max_iter=1, random_state=42)
    
    kmeans.fit(data)
    return kmeans.labels_, kmeans.cluster_centers_

def plot_current_state(data, labels, centroids):
    """Plot the current state of clustering."""
    fig = px.scatter(x=data[:, 0], y=data[:, 1], color=labels,
                     color_continuous_scale=px.colors.qualitative.Set1,
                     labels={'x': 'Feature 1', 'y': 'Feature 2'},
                     title='K-Means Clustering')
    fig.add_scatter(x=centroids[:, 0], y=centroids[:, 1], mode='markers',
                    marker=dict(color='black', size=12, symbol='star'),
                    name='Centroids')
    return fig

# Streamlit app
st.title('K-Means Clustering Visualizer')

# Sidebar for user inputs
st.sidebar.header('Parameters')

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

data_source = st.sidebar.radio('Data Source', ['Generate Random', 'Upload CSV'])

if data_source == 'Generate Random':
    n_samples = st.sidebar.slider('Number of data points', 50, 1000, 200)
    n_features = st.sidebar.slider('Number of features', 2, 3, 2)
    
    if 'data' not in st.session_state or st.sidebar.button('Generate New Data'):
        st.session_state.data = generate_random_data(n_samples, n_features)
        st.session_state.iteration = 0
else:
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.session_state.data = data.values
        st.session_state.iteration = 0
    elif 'data' not in st.session_state:
        st.warning("Please upload a CSV file or switch to 'Generate Random' data.")
        st.stop()

data = st.session_state.data

n_clusters = st.sidebar.slider('Number of clusters (K)', 2, 10, 3)
max_iter = st.sidebar.slider('Max iterations', 1, 20, 10)

# Initialize or reset K-Means
if 'kmeans' not in st.session_state or st.sidebar.button('Reset K-Means'):
    st.session_state.kmeans = KMeans(n_clusters=n_clusters, max_iter=1, n_init=1, random_state=42)
    st.session_state.labels = np.zeros(data.shape[0], dtype=int)  # Initialize labels
    # Initialize centroids randomly
    st.session_state.centroids = data[np.random.choice(data.shape[0], n_clusters, replace=False)]
    st.session_state.iteration = 0

# Run K-Means step
if st.button('Run K-Means Step'):
    if st.session_state.iteration < max_iter:
        st.session_state.labels = st.session_state.kmeans.fit_predict(data)
        st.session_state.centroids = st.session_state.kmeans.cluster_centers_
        st.session_state.iteration += 1
    else:
        st.warning('Maximum iterations reached. Reset K-Means to start over.')

# Display current state
st.write(f'Current iteration: {st.session_state.iteration}')

# Plot the current state
if 'kmeans' in st.session_state:
    labels = st.session_state.labels
    centroids = st.session_state.centroids
    
    try:
        fig = plot_clusters(data, labels, centroids)
        st.plotly_chart(fig)
    except ValueError as e:
        st.warning(str(e))
        st.warning("Visualization is only supported for 2D and 3D data.")

# Elbow method
if st.checkbox('Show Elbow Method'):
    max_k = st.slider('Maximum K for Elbow Method', 2, 15, 10)
    inertias = elbow_method(data, max_k)
    elbow_fig = plot_elbow(inertias)
    st.plotly_chart(elbow_fig)

# Silhouette score
if st.checkbox('Show Silhouette Score'):
    max_k = st.slider('Maximum K for Silhouette Score', 2, 15, 10)
    silhouette_fig = plot_silhouette(data, max_k)
    st.plotly_chart(silhouette_fig)

# Animation
if st.checkbox('Show Clustering Animation'):
    if data.shape[1] == 2:  # Only show animation for 2D data
        n_clusters = st.sidebar.slider('Number of clusters (K) for animation', 2, 10, 3, key='n_clusters_animation')
        max_iter = st.sidebar.slider('Max iterations for animation', 1, 20, 10, key='max_iter_animation')
        
        # Initialize centroids
        centroids = None
        
        # Create a placeholder for the plot
        plot_placeholder = st.empty()
        
        # Create a button to start/reset the animation
        if st.button('Start/Reset Animation'):
            for i in range(max_iter):
                labels, centroids = run_kmeans_step(data, n_clusters, centroids)
                fig = plot_current_state(data, labels, centroids)
                plot_placeholder.plotly_chart(fig, use_container_width=True)
                time.sleep(0.5)  # Pause for half a second between iterations
            
            st.success("Animation complete!")
    else:
        st.warning("Animation is only supported for 2D data.")

# Display data table
if st.checkbox('Show data table'):
    df = pd.DataFrame(data, columns=[f'Feature {i+1}' for i in range(data.shape[1])])
    df['Cluster'] = st.session_state.labels
    st.write(df)

# Explanation
st.markdown("""
## How K-Means Clustering Works

1. **Initialization**: K cluster centroids are randomly initialized.
2. **Assignment**: Each data point is assigned to the nearest centroid.
3. **Update**: Centroids are recalculated as the mean of all points in that cluster.
4. **Repeat**: Steps 2 and 3 are repeated until convergence or max iterations.

This visualizer allows you to see this process step by step. Generate new data or upload your own, 
adjust the number of clusters, and run the algorithm to see how it works!

### Features

1. **Elbow Method**: Helps in finding the optimal number of clusters by plotting the inertia against K.
2. **Silhouette Score**: Another method to evaluate the quality of clustering for different K values.
3. **Custom Dataset Upload**: You can now upload your own CSV file for clustering.
4. **3D Visualization**: If your data has 3 features, you can visualize the clustering in 3D.
5. **Clustering Animation**: Watch how the clusters evolve over iterations.

These tools can help in understanding the K-Means algorithm better and in choosing the right number of clusters for your data.
""")