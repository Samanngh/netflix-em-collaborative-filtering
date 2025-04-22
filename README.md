# Netflix EM Collaborative Filtering

This project implements an Expectation-Maximization (EM) algorithm for collaborative filtering using a Gaussian Mixture Model (GMM), applied to a Netflix-style movie rating dataset with missing entries.

## 📌 Project Goals

- Impute missing ratings using probabilistic matrix factorization
- Compare K-Means and EM clustering for unsupervised learning
- Evaluate log-likelihood and RMSE for model selection

## 📁 Project Structure

```
.
├── em.py                # Core EM algorithm implementation
├── naive_em.py          # Simplified EM (no missing data)
├── kmeans.py            # K-Means algorithm used for initialization
├── common.py            # Utility functions, GaussianMixture class
├── main.py              # Main script to run experiments and evaluate BIC
├── test.py              # RMSE testing script against gold data
├── netflix_incomplete.txt  # Dataset with missing entries
├── netflix_complete.txt    # Ground truth for evaluation
├── toy_data.txt         # 2D toy dataset for visualizations
└── test_solutions.txt   # Reference results for grading
```

## 🔍 Highlights

- **Matrix Completion** using EM with Gaussian Mixtures
- **Numerical Stability** ensured with logsumexp
- **Soft Assignments** using responsibilities (posterior probabilities)
- **Model Selection** with Bayesian Information Criterion (BIC)
- **RMSE Reporting** for both missing entries and full dataset

## ✅ Sample Results

```
Best Log-Likelihood for K=12: -1390234.42
RMSE on missing entries: 1.0064
RMSE on all entries:      0.4805
```

## 🚀 How to Run

```bash
# Clone the repository
$ git clone https://github.com/Samanngh/netflix-em-collaborative-filtering.git
$ cd netflix-em-collaborative-filtering

# Run the full experiment
$ python main.py

# Test the imputation accuracy
$ python test.py
```

## 📚 Dependencies

- Python 3.7+
- NumPy
- Matplotlib
- SciPy
