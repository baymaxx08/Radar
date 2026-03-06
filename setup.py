from setuptools import setup, find_packages

setup(
    name="corn-cob-classifier",
    version="1.0.0",
    description="Corn Cob Classification System using Neural Networks",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "flask==3.0.0",
        "flask-cors==4.0.0",
        "tensorflow==2.10.0",
        "numpy==1.23.5",
        "pandas==1.5.3",
        "scikit-learn==1.2.2",
        "python-dotenv==1.0.0",
        "gunicorn==21.2.0",
        "Werkzeug==2.2.3",
        "protobuf==3.20.0",
        "h5py==3.8.0",
    ],
)
