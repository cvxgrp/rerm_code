import numpy as np
import matplotlib.pyplot as plt
from data_processing import read_and_filter_data, extract_train_df_test_df, mask_train_df, create_test_arr, create_local_radius_centers_diags, find_city_center, create_diag_matrix
from solver import solve_robust_prob
from evaluate import mean_squared_error


"""Longitude Always Comes Before Latitude"""
train_test_rng = np.random.default_rng(0)
num_train = 1000
num_test = 500
masked_features = ["lng", "lat"]
city_name = "london"
columns_to_drop = ["Unnamed: 0", "room_type", "room_shared", "room_private", "host_is_superhost",
                   "attr_index", "rest_index", "multi", "biz", "bedrooms"]

filtered_df = read_and_filter_data(
    city_name=city_name,
    weekdays=True,
    filter_price_upper=1000,
    filter_price_lower=0,
    filter_dist_upper=7,
    columns_to_drop=columns_to_drop
)

train_df, test_df = extract_train_df_test_df(
    df=filtered_df, 
    num_train=num_train,
    num_test=num_test,
    rng=train_test_rng
)

test_arr, test_prices = create_test_arr(test_df.drop(columns=["dist"]))

# this is half the side length of the square that contains the location
local_radius_const = 0.5
mask_prob_arr = np.array([0,1.0])

results = []
rng = np.random.default_rng(0)

# solve problem with all latitude and longitude data present
# i.e., no location feature information is masked
mask_prob = 0.0
masked_train_arr, train_prices, unmasked_train_df = mask_train_df(train_df, mask_prob, masked_features, rng)

city_center = find_city_center(city=city_name)
city_center_diag_matrix = create_diag_matrix(center=city_center)

pred, intercept = solve_robust_prob(
        A=masked_train_arr,
        b=train_prices,
        unmasked_A_df=unmasked_train_df,
        use_city_center_constraint=False,
        city_center=city_center,
        city_center_diag_matrix=city_center_diag_matrix
    )
ols_test_error = mean_squared_error(pred=pred, intercept=intercept, feature_matrix=test_arr,
                                response_vector=test_prices)
print(ols_test_error)

# solve problem with no latitude and longitude data present
# i.e., all location feature information is masked
# solve with the two robust predictors

mask_prob = 1.0
masked_train_arr, train_prices, unmasked_train_df = mask_train_df(train_df, mask_prob, masked_features, rng)

city_center = find_city_center(city=city_name)
city_center_diag_matrix = create_diag_matrix(center=city_center)

# square only uncertainty sets
pred, intercept = solve_robust_prob(
    A=masked_train_arr,
    b=train_prices,
    unmasked_A_df=unmasked_train_df,
    use_city_center_constraint=False,
    city_center=city_center,
    city_center_diag_matrix=city_center_diag_matrix,
    use_grid=create_local_radius_centers_diags(masked_train_arr, unmasked_train_df, local_radius_const, rng, grid = True)
)
square_only_test_error = mean_squared_error(pred=pred, intercept=intercept, feature_matrix=test_arr,
                                response_vector=test_prices)
results.append((square_only_test_error - ols_test_error) / ols_test_error * 100)
print(square_only_test_error)

# square and circular uncertainty sets
pred, intercept = solve_robust_prob(
    A=masked_train_arr,
    b=train_prices,
    unmasked_A_df=unmasked_train_df,
    use_city_center_constraint=True,
    city_center=city_center,
    city_center_diag_matrix=city_center_diag_matrix,
    use_grid = create_local_radius_centers_diags(masked_train_arr, unmasked_train_df, local_radius_const, rng, grid=True)
)
square_and_circle_test_error = mean_squared_error(pred=pred, intercept=intercept, feature_matrix=test_arr,
                                response_vector=test_prices)
results.append((square_and_circle_test_error - ols_test_error) / ols_test_error * 100)
print(square_and_circle_test_error)

# drop latitude and longitude features, no robustness
dropped_arr = masked_train_arr[:, :-2]
assert True not in np.isnan(dropped_arr)
assert np.shape(dropped_arr)[1] == np.shape(masked_train_arr)[1] - 2
assert np.shape(dropped_arr)[0] == np.shape(masked_train_arr)[0]
pred, intercept = solve_robust_prob(
    A=dropped_arr,
    b=train_prices,
    unmasked_A_df=unmasked_train_df,
    use_city_center_constraint=False,
    city_center=city_center,
    city_center_diag_matrix=city_center_diag_matrix
)
pred_with_zeros = [i for i in pred]
pred_with_zeros.append(0)
pred_with_zeros.append(0)
pred_with_zeros = np.array(pred_with_zeros)
train_error = mean_squared_error(pred=pred_with_zeros, intercept=intercept, feature_matrix=np.array(unmasked_train_df.drop(columns=["dist"])),
                                 response_vector=train_prices)
dropped_test_error = mean_squared_error(pred=pred_with_zeros, intercept=intercept, feature_matrix=test_arr,
                                response_vector=test_prices)
results.append((dropped_test_error - ols_test_error) / ols_test_error * 100)
print(dropped_test_error)

print(results)

plt.close('all')
plt.bar(["S", r"$D \cap S$", "Drop"], results)
plt.ylabel("Percentage increase in test error over OLS")
plt.show()