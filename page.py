import streamlit as st

def main():
    st.set_page_config(page_title="ML Library Visualizer", layout="wide")

    st.title("Welcome to ML Library Visualizer")

    # Add LinkedIn and GitHub links
    linkedin_logo = "https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg"
    github_logo = "https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"

    st.markdown(f"""
    <div style="display: flex; align-items: center; margin-bottom: 20px;">
        <a href="https://www.linkedin.com/in/shaikhmubin/" target="_blank" style="margin-right: 20px;">
            <img src="{linkedin_logo}" width="30" alt="LinkedIn">
        </a>
        <a href="https://github.com/shaikhmubin02/ml-library-visualizer" target="_blank">
            <img src="{github_logo}" width="30" alt="GitHub">
        </a>
    </div>
    """, unsafe_allow_html=True)

    st.write("""
    This application provides interactive visualizations for machine learning algorithms. 
    Currently, available visualizations include:
    
    - K-means Clustering
    - Neural Networks
    
    Click on a button below to navigate to the corresponding visualizer.
    """)

    # CSS for the buttons
    button_style = """
    <style>
    .custom-button {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 200px;
        padding: 0.3rem 0.5rem;
        margin: 0.25rem 0;
        border-radius: 2rem;
        background-color: #ffffff;
        color: #000000;
        font-size: 1rem;
        font-weight: 400;
        line-height: 1.5;
        text-align: center;
        text-decoration: none;
        cursor: pointer;
        border: 1px solid #ccc;
        box-shadow: rgba(0, 0, 0, 0.05) 0px 1px 2px 0px;
        transition: all 0.2s ease-in-out;
    }
    .custom-button:hover {
        background-color: #f0f2f6;
        color: #000000;
        border-color: #d2d2d2;
    }
    .custom-button:active {
        background-color: #e0e0e0;
        border-color: #a0a0a0;
        color: #000000;
    }
    </style>
    """

    # HTML for the buttons
    buttons_html = f"""
    {button_style}
    <div style="display: flex; justify-content: space-around;">
        <a href="https://k-means-visualizer.streamlit.app" class="custom-button">K-means Visualizer</a>
        <a href="https://nn-visualizer.streamlit.app/" class="custom-button">Neural Network Visualizer</a>
    </div>
    """

    st.markdown(buttons_html, unsafe_allow_html=True)

    st.subheader("About the Project")
    st.write("""
    ML Library Visualizer is an educational tool designed to help users understand 
    the inner workings of various machine learning algorithms through interactive 
    visualizations. To contribute or explore the project, visit the GitHub repository.
    """)

    st.subheader("How to Use")
    st.write("""
    1. Click on one of the visualizer buttons above.
    2. Follow the instructions on each visualizer page to interact with the algorithm.
    3. Observe how changes in parameters affect the algorithm's behavior and output.
    """)

if __name__ == "__main__":
    main()