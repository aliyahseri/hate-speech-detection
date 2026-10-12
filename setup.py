from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hatespeech-detection",
    version="0.1.0",
    author="Aliyah Seri",
    description="A deep learning-based hate speech detection system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aliyahseri/hate-speech-detection",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "torch>=1.9.0",
        "transformers>=4.11.0",
        "scikit-learn>=0.24.2",
        "tqdm>=4.62.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "nltk>=3.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.7b0",
            "isort>=5.9.0",
            "mypy>=0.910",
            "flake8>=3.9.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hatespeech-train=src.train:main",
            "hatespeech-eval=src.evaluate:main",
            "hatespeech-infer=src.inference:main",
        ],
    },
)
