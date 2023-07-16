# Agricultural Water Consumption Optimization (Hackathon)

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Overview
This project aims to optimize water consumption in agriculture by developing an AI solution based on the agricultural dataset for Rajasthan, India, from the Kaggle dataset repository. The solution uses machine learning techniques to predict and optimize water usage based on soil analysis and crop production data.

## Dataset
The dataset used in this project is the [Agricultural Data for Rajasthan, India (2018-2019)](https://www.kaggle.com/datasets/suraj520/agricultural-data-for-rajasthan-india-2018-2019). It includes information on crop production, soil analysis, and water usage. The dataset has been preprocessed and merged for the analysis.

## Requirements
- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- XGBoost  
- Category Encoders  
- Missingno  

Install the required packages using pip:

```pip install -r requirements.txt```

## Project Structure
The project repository is organized as follows:

- `data/`: Contains the dataset files (crop production, soil analysis, and water usage).
- `notebooks/`: Jupyter notebooks for data exploration, preprocessing, and modeling.
- `src/`: Source code files for the project.
    - `data_preprocessing.py`: Implements functions for loading, preprocessing, and merging the dataset.
    - `modeling.py`: Contains the machine learning models and functions for training and evaluation.
    - `utils.py`: Utility functions used in the project.
- `results/`: Contains the model evaluation results and visualizations.
- `README.md`: Project overview, instructions, and documentation.

## Usage
1. Clone the repository:

git clone https://github.com/your-username/agricultural-water-consumption.git
cd agricultural-water-consumption

2. Install the required packages:

pip install -r requirements.txt


3. Run the Jupyter notebooks in the `notebooks/` directory for data exploration, preprocessing, and modeling.

4. Execute the main script to train and evaluate the machine learning models:

python src/main.py


## Results
The results of the project, including evaluation metrics and visualizations, are stored in the `results/` directory. The findings and insights can be found in the [project report](report/report.pdf).

## License
This project is licensed under the [MIT License](LICENSE).

## Contributors
- John Doe [@johndoe](https://github.com/johndoe)
- Jane Smith [@janesmith](https://github.com/janesmith)

## Acknowledgements
We would like to thank the contributors of the [Agricultural Data for Rajasthan, India (2018-2019)](https://www.kaggle.com/datasets/suraj520/agricultural-data-for-rajasthan-india-2018-2019) dataset on Kaggle.
