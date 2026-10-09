# Car Price Prediction using Multiple Linear Regression

Predicts the price of a car from its numeric specifications using linear regression implemented **from scratch** with NumPy and gradient descent. No scikit-learn or other ML library is used.

## Files
- `Car_Price_Multiple_Regression.ipynb` - the notebook with all the code and outputs
- `CarPrice_Assignment.csv` - the dataset (205 cars)

## Dataset
Car Price Prediction dataset (source: UCI Automobile dataset). `price` is the target. The 14 numeric feature columns are used: `symboling`, `wheelbase`, `carlength`, `carwidth`, `carheight`, `curbweight`, `enginesize`, `boreratio`, `stroke`, `compressionratio`, `horsepower`, `peakrpm`, `citympg`, `highwaympg`. `car_ID` (identifier) and the text columns are not used.

## Workflow
1. **Load** the data.
2. **Split** into 80% training (164 cars) and 20% testing (41 cars), shuffled with a fixed seed.
3. **Normalize** each feature as `(x - mean) / (max - min)`, using the training set's mean and range only (also applied to the test set, so nothing leaks from the test data). A bias column of ones is added.
4. **Train** with gradient descent on the training set only (5000 iterations, learning rate 0.1, weights start at zero).
5. **Predict** prices for the unseen test cars.
6. **Evaluate** with cost, R², RMSE and MAE on both sets, and visualize.

## Model
- Prediction: `predictions = features . weights`
- Cost (MSE): `1/(2N) * sum((predictions - targets)^2)`
- Gradient descent: `weights -= learning_rate * gradient`, where `gradient = -features.T . (targets - predictions) / N`


## Results

| Metric | Training | Testing |
|---|---|---|
| R² | 0.855 | 0.804 |
| RMSE | 2943 | 3895 |
| MAE | 2096 | 2890 |

On cars it has never seen, the model explains about 80% of the variation in price, with an average error of roughly 2,900. Test performance is a little lower than training, as expected, and the test set is only 41 cars, so the numbers would shift somewhat with a different split.

## Visualizations in the notebook
- Training cost history
- Test set: actual vs predicted price
- Test set residuals (vs predicted, and distribution)
- Learned feature weights
