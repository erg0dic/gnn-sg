from setuptools import setup, find_packages
from pathlib import Path

here = Path(__file__).parent.resolve()

# Get the long description from the README file
LONG_DESC = (here / "README.md").read_text(encoding="utf-8")
VERSION = "0.0.0"
setup(
    name="gnnsg",  # (Graph Neural Network For Systematic Generalisation) Required 
    version=VERSION,  # (Major.Minor.Maintenance) Required
    description="Graph Neural Networks For Systematic Generalisation", 
    long_description=LONG_DESC,
    long_description_content_type="text/markdown",
    author="Anonymous",  # Optional
    classifiers=[ 
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Researchers",
        "Intended Audience :: Developers",
        "Topic :: Graph Neural Networks :: Systematic Generalisation",
        "Topic :: Scientific/Research :: Mathematics",
        "Topic :: Scientific/Research :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3"
    ],
    keywords="graph neural network, link prediction, systematic generalisation, neurosymbolic, theorem prover",  # Optional
    packages=find_packages(),  # Required
    install_requires=[
                      "torch==2.1.2",
                      "torch_geometric",
                      "torchvision",
                      "torchaudio",
                      "numpy",
                      "hydra-core",
                      "wandb", 
                      "pytest", 
                      "lightning",
                      "transformers",
                      "matplotlib", 
                      "regex"],  # Optional
    python_requires=">=3.8",
    # dependency_links=["https://data.pyg.org/whl/torch-2.1.0+cu121.html", 
    #                   ],        # This should be consistent with the installed CUDA version on your machine
    # additional groups of dependencies here using the "extras"
    extras_require={  # Optional, NOTE: these are one-way requirements, iirc, and only flag if used... so just TODO for now
        "dev": ["check-manifest"],
        "in-square-brackets-test": ["wandb"] # test
    },
    # Entry points. Alias commands for script functions can just be called from CLI without saying "python [fname].py"
    entry_points={  # Optional
        "console_scripts": [
            "run=src.train:run",
        ],
    },
)