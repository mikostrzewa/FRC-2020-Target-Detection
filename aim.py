# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

while True:
	# Importing the dataset
	dataset = pd.read_csv('data.csv')
	X = dataset.iloc[:, 0:1].values
	y = dataset.iloc[:, 1].values

	from sklearn.linear_model import LinearRegression
	from sklearn.preprocessing import PolynomialFeatures
	poly_reg = PolynomialFeatures(degree = 3)
	X_poly = poly_reg.fit_transform(X)
	poly_reg.fit(X_poly, y)
	lin_reg_2 = LinearRegression()
	lin_reg_2.fit(X_poly, y)

	# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
	X_grid = np.arange(min(X), max(X), 0.1)
	X_grid = X_grid.reshape((len(X_grid), 1))
	plt.scatter(X, y, color = 'red')
	plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
	plt.title('Team Rabyte 8127 Polynomial Regression)')
	plt.xlabel('Distance')
	plt.ylabel('Angle')
	plt.show(block=False)
	plt.pause(0.0001)
	plt.close()
	#To retrieve the intercept:
	print(lin_reg_2.intercept_)
	#For retrieving the slope:
	print(lin_reg_2.coef_)


