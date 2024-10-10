# Machine Learning in C++
## About
The goal of this project is to recreate the simplicity of Python's Tensorflow and numpy in C++. This library implements core ndarray and related math operations alongside a barebones autograd engine to get the power of a Keras-like library

## Example usage
```C++
// Our independent and dependent variables
ml::matrix_t y;
ml::matrix_t X;

load_data(X, y); // Your function to load data from a file into the matricies

// Setup a logistic regression
ml::regression::logistic logreg(y, X);

// Run an SGD optimization with cross-entropy loss
SGD sgd(ml::metrics::cross_entropy, 0.0001, 100);
sgd.optimize(logreg, {X}, y);
```
