from setuptools import setup, find_packages

setup(
    name="dacon",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "scikit-learn",
        "tqdm",
        "transformers",
        "pillow",
        "opencv-python",
    ],
    python_requires=">=3.8",
) 