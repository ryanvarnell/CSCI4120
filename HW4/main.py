# Ryan Varnell
# CSCI 4120

import numpy as np
import pandas as pd
# For Jupyter:
# %matplotlib inline
import seaborn as sns
from pandas.tseries.holiday import USFederalHolidayCalendar
from scipy.stats import uniform
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

sns.set()

counts = pd.read_csv('data/FremontBridge.csv', index_col='Date', parse_dates=True)
weather = pd.read_csv('data/BicycleWeather.csv', index_col='DATE', parse_dates=True)

# Compute daily traffic
daily = counts.resample('d').sum()
daily['Total'] = daily.sum(axis=1)
daily = daily[['Total']]  # remove other columns

# Add day indicator
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
for i in range(7):
    daily[days[i]] = (daily.index.dayofweek == i).astype(float)

# Account for holidays
cal = USFederalHolidayCalendar()
holidays = cal.holidays('2012', '2016')
daily = daily.join(pd.Series(1, index=holidays, name='holiday'))
# replace missing data with 0
daily['holiday'].fillna(0, inplace=True)


# Compute the hours of daylight for the given date
def hours_of_daylight(date, axis=23.44, latitude=47.61):
    """Compute the hours of daylight for the given date"""
    days = (date - pd.datetime(2000, 12, 21)).days
    m = (1. - np.tan(np.radians(latitude))
         * np.tan(np.radians(axis) * np.cos(days * 2 * np.pi / 365.25)))
    return 24. * np.degrees(np.arccos(1 - np.clip(m, 0, 2))) / 180.


daily['daylight_hrs'] = list(map(hours_of_daylight, daily.index))

# temperatures are in 1/10 deg C; convert to C
weather['TMIN'] /= 10
weather['TMAX'] /= 10
weather['Temp (C)'] = 0.5 * (weather['TMIN'] + weather['TMAX'])

# precip is in 1/10 mm; convert to inches
weather['PRCP'] /= 254
weather['dry day'] = (weather['PRCP'] == 0).astype(int)

daily = daily.join(weather[['PRCP', 'Temp (C)', 'dry day']])

# Calculate number of years passed
daily['annual'] = (daily.index - daily.index[0]).days / 365.

# Drop rows with null values
daily.dropna(axis=0, how='any', inplace=True)

column_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'holiday',
                'daylight_hrs', 'PRCP', 'dry day', 'Temp (C)', 'annual']
X = daily[column_names]
y = daily['Total']

# Fit to LinearRegression
lr_model = LinearRegression(fit_intercept=False)
lr_model.fit(X, y)
lr_cv_score = cross_val_score(lr_model, X, y, cv=10)

# Tune alpha and fit to Ridge
ridge_model = Ridge()
ridge_parameters = {'alpha': uniform()}
ridge_cv = RandomizedSearchCV(ridge_model, ridge_parameters, n_iter=100, cv=10, n_jobs=-1)
ridge_search = ridge_cv.fit(X, y)
ridge_alpha = ridge_search.best_estimator_.alpha
ridge_model = Ridge(alpha=ridge_alpha)
ridge_model.fit(X, y)
ridge_cv_score = cross_val_score(ridge_model, X, y, cv=10)

# Tune lasso and fit to Lasso
lasso_model = Lasso()
lasso_parameters = {'alpha': uniform()}
lasso_cv = RandomizedSearchCV(lasso_model, lasso_parameters, n_iter=100, cv=10, n_jobs=-1)
lasso_search = lasso_cv.fit(X, y)
lasso_alpha = lasso_search.best_estimator_.alpha
lasso_model = Lasso(alpha=lasso_alpha)
lasso_model.fit(X, y)
lasso_cv_score = cross_val_score(lasso_model, X, y, cv=10)

print("Linear Regression cross val scores: " + str(lr_cv_score) + "\nMean: " + str(np.mean(lr_cv_score)) + "\n")
print("Ridge alpha: " + str(ridge_alpha))
print("Ridge cross val scores: " + str(ridge_cv_score) + "\nMean: " + str(np.mean(ridge_cv_score)) + "\n")
print("Lasso alpha: " + str(lasso_alpha))
print("Lasso cross val scores: " + str(lasso_cv_score) + "\nMean: " + str(np.mean(lasso_cv_score)) + "\n")
