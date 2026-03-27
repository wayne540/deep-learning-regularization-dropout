# Regularization in Deep Neural Networks

This repository contains my implementation of regularization techniques from **Course 2: Improving Deep Neural Networks** (Coursera Deep Learning Specialization).

The goal of this project is to understand and implement methods that reduce overfitting in neural networks, specifically:

* L2 Regularization
* Dropout

---

##  Problem Overview

Given a 2D dataset representing positions on a football field, the objective is to train a neural network that predicts where a goalkeeper should kick the ball so that their team is more likely to win aerial duels.

* **Blue points (1):** Successful header by the home team
* **Red points (0):** Opponent wins the header

The dataset is noisy, making it prone to **overfitting**, which makes it ideal for testing regularization techniques.

---

##  Model Architecture

The neural network used in this project follows a 3-layer architecture:

```
LINEAR → RELU → LINEAR → RELU → LINEAR → SIGMOID
```

Layer dimensions:

```
[input → 20 → 3 → 1]
```

---

##  Features Implemented

### 1. Baseline Model (No Regularization)

* Standard forward and backward propagation
* Cross-entropy loss
* Gradient descent optimization

---

### 2. L2 Regularization

#### Cost Function Modification:

The cost function is extended with a penalty term:

```
J_regularized = J + (λ / 2m) * Σ ||W||²
```

#### Effect:

* Penalizes large weights
* Encourages smoother decision boundaries
* Reduces overfitting

#### Implementation:

* `compute_cost_with_regularization`
* `backward_propagation_with_regularization`

---

### 3. Dropout Regularization

Dropout randomly deactivates neurons during training.

#### Key Steps:

1. Create a dropout mask `D`
2. Apply mask to activations
3. Scale activations by `keep_prob`

#### Important:

* Applied only during training
* Disabled during testing

#### Implementation:

* `forward_propagation_with_dropout`
* `backward_propagation_with_dropout`

---

##  Results

| Model                       | Train Accuracy | Test Accuracy |
| --------------------------- | -------------- | ------------- |
| No Regularization           | 95%            | 91.5%         |
| L2 Regularization (λ = 0.7) | 94%            | 93%           |
| Dropout (keep_prob = 0.86)  | 93%            | 95%           |

---

##  Key Observations

* The baseline model overfits the training data.
* L2 regularization smooths the decision boundary.
* Dropout improves generalization the most.
* Regularization reduces training accuracy but improves test performance.

---

##  How to Run

1. Clone the repository:

```
git clone <your-repo-link>
cd <repo-folder>
```

2. Install dependencies:

```
pip install numpy matplotlib scipy scikit-learn h5py
```

3. Run the notebook or script:

```
python your_script.py
```

---

##  Project Structure

```
.
├── reg_utils.py        # Core utility functions (forward/backward propagation, etc.)
├── testCases.py        # Provided test cases
├── data.mat            # Dataset file
├── train_catvnoncat.h5 # Optional dataset
├── test_catvnoncat.h5  # Optional dataset
└── README.md
```

---

##  Key Takeaways

* Regularization is essential for preventing overfitting.
* L2 regularization shrinks weights (weight decay).
* Dropout forces the network to learn more robust features.
* Always evaluate on a validation/test set, not just training data.

---

##  References

* Coursera Deep Learning Specialization
* Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization

---

##  Author

Developed as part of my deep learning journey, focusing on building strong fundamentals in neural networks and optimization techniques.
