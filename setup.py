from setuptools import setup, find_packages

setup(
    name='drugs4disease',
    version='0.1.0',
    description='A tool for using NM-based model to predict drugs for a given disease',
    author='Jingcheng Yang',
    author_email='yjcyxky@163.com',
    packages=find_packages(),
    install_requires=[
        'click>=8.0.0',
        'pandas>=1.3.0',
        'numpy>=1.21.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0',
        'networkx>=2.6.0',
        'gseapy>=0.12.0',
        'gprofiler-official>=1.0.0',
        'torch>=1.9.0',
        'openpyxl>=3.0.0',
        'requests>=2.25.0',
        'scikit-learn>=1.0.0',
        'wandb>=0.15.0',
    ],
    entry_points={
        'console_scripts': [
            'drugs4disease=drugs4disease.cli:cli',
        ],
    },
    python_requires='>=3.8',
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
) 