# MLOps - Module 1

This repository contains the implementation of a CI/CD pipeline for a sample machine learning project as part of the MLOps assignment. The pipeline includes stages for linting, testing, and deploying a simple machine learning model.

## **Project Overview**
- **Objective**: Implement a CI/CD pipeline for a machine learning project using GitHub Actions.
- **Pipeline Stages**:
  1. Linting
  2. Testing
  3. Deployment (Simulated)
- **Tools Used**:
  - Python
  - GitHub Actions for CI/CD
  - Flake8 for linting
  - Pytest for testing

---

## **Setup Instructions**

### Prerequisites
1. Install [Python 3.9+](https://www.python.org/downloads/).
2. Install [Git](https://git-scm.com/).
3. Create a GitHub account (if not already available).
4. Clone this repository:
   ```bash
   git clone <repository-link>
   cd <repository-folder>
   ```

---

### Step 1: Install Dependencies

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/Mac
   venv\Scripts\activate   # For Windows
   ```

2. Install project dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

### Step 2: Run the Machine Learning Model

1. Navigate to the main project folder.
2. Execute the model training script:
   ```bash
   python train_model.py
   ```
   This script trains a sample machine learning model and exports it to the `models/` directory.

---

### Step 3: Set Up CI/CD Pipeline

1. Ensure your repository is hosted on GitHub.
2. Add the CI/CD workflow file:
   - Navigate to `.github/workflows/main.yml`.
   - Verify the workflow stages (Lint, Test, Deploy).
3. Commit and push changes to trigger the pipeline:
   ```bash
   git add .
   git commit -m "Added CI/CD pipeline"
   git push origin main
   ```

---

### Step 4: Version Control

1. Create a new branch for development:
   ```bash
   git checkout -b feature/new-feature
   ```

2. Commit your changes:
   ```bash
   git commit -m "Description of changes"
   ```

3. Push the branch and create a pull request:
   ```bash
   git push origin feature/new-feature
   ```

4. Merge the pull request into the `main` branch.

---

## **Pipeline Details**

### CI/CD Workflow Stages
1. **Linting**:
   - Tool: `flake8`
   - Command: `flake8 .`

2. **Testing**:
   - Tool: `pytest`
   - Command: `pytest`

3. **Deployment**:
   - Simulated by a simple shell script echoing deployment success.

---

## **Project Structure**
```
.
├── .github/workflows/main.yml   # CI/CD Pipeline configuration
├── models/                      # Trained models directory
├── src/                         # Source code for the project
│   ├── train_model.py           # Script to train the ML model
│   └── utils.py                 # Utility functions
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## **Sample Commands**

### Run Linting Locally
```bash
flake8 .
```

### Run Tests Locally
```bash
pytest
```

---

## **Contribution Guidelines**
1. Fork the repository.
2. Create a feature branch.
3. Commit and push changes.
4. Open a pull request for review.

---

## **License**
This project is licensed under the MIT License. See the LICENSE file for details.

---

## **Contact**
For any issues or queries, please create an issue in this repository or contact the author at [your-email@example.com].
