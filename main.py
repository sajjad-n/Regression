from sklearn import linear_model
import pandas

# reading data
train_set = pandas.read_csv('TrainSet1.csv')
test_set = pandas.read_csv('TestSet1.csv')

# selecting columns
year_train = train_set.drop(columns=['population'])
population_train = train_set.drop(columns=['year'])

# linear regression
linear = linear_model.LinearRegression()

# train model
linear.fit(year_train, population_train)

# predict
predicted_population = linear.predict(test_set)

# show result
predict_years_number = len(test_set)
print('predict for', predict_years_number, 'years:')
for i in range(0, predict_years_number):
    print('year:', int(test_set.values[i]), ' population:', int(predicted_population[i]))
