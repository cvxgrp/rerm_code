import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List
from math import ceil, floor

def read_and_filter_data(
        city_name: str,
        weekdays: bool,
        filter_price_upper: float,
        filter_price_lower: float,
        filter_dist_upper: float,
        columns_to_drop: List[str]
) -> pd.DataFrame:
    """
    Reads the data in the CSV and returns a dataframe that has no missing entries, and is appropriately
    filtered.

    :param city_name: the string containing the city
    :param weekdays: if True, then we use the weekdays file, otherwise the weekends file
    :param filter_price_upper: filter the dataset to use this as an upper bound on realSum (price)
    :param filter_price_lower: filter the dataset to use this as an lower bound on realSum (price)
    :param filter_dist_upper: filter the dataset to use this as an upper bound on dist (distance from city center)
    :param columns_to_drop: a List of columns that should be dropped
    
    :return: a Dataframe satsfying the above properties
    """
    if weekdays:
        filename = "data_files/" + city_name + "_weekdays.csv"
    else:
        filename = "data_files/" + city_name + "_weekends.csv"
    total_df = pd.read_csv(filename)

    # make sure there are no missing entries in the original dataset
    print("Verifying Original Dataset Has No Missing Entries")
    assert total_df.isnull().values.any() == False
    
    print("Original Dataset Size: " + str(np.shape(total_df)))
    total_df = total_df.drop(columns=columns_to_drop)

    print("Pre-Filtered Dataset Size: " + str(np.shape(total_df)))
    total_df = total_df[total_df["realSum"] <= filter_price_upper]
    total_df = total_df[total_df["realSum"] >= filter_price_lower]
    total_df = total_df[total_df["dist"] <= filter_dist_upper]
    print("Post-Filtered Dataset Size: " + str(np.shape(total_df)))

    return total_df

def extract_train_df_test_df(
        df: pd.DataFrame,
        num_train: int,
        num_test: int,
        rng: np.random.Generator
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    The dataframe whose rows we will sample. Assume it has n rows. Then we sample num_train rows
    that are used for training set, and sample a DIFFERENT num_test rows for test set.

    :param df: The input dataframe
    :param num_train: the number of rows that will be in the training set
    :param num_test: the number of rows that will be in the test set
    :param rng: The random generator object used to create the random indices
    :return: A tuple where the first entry is the training df and the second entry is the test df
    """
    n = len(df)
    all_row_indices = np.arange(n)
    training_row_indices = rng.choice(all_row_indices, size=num_train, replace=False)
    possible_test_row_indices = all_row_indices[~np.isin(all_row_indices, training_row_indices)]
    test_row_indices = rng.choice(possible_test_row_indices, size=num_test, replace=False)
    train_df = df.iloc[training_row_indices]
    test_df = df.iloc[test_row_indices]

    assert len(training_row_indices) == num_train
    assert len(test_row_indices) == num_test
    for i in training_row_indices:
        assert i not in test_row_indices
    return train_df, test_df

def mask_train_df(
        train_df: pd.DataFrame,
        mask_prob: float,
        feature_names: List[str],
        rng: np.random.Generator
)-> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each row in df, we independently sample a Bernoulli with parameter mask_prob. If it is 1, then
    we hide the entries in the columns lying in feature_names to be Nans

    :param train_df: a Dataframe whose entries we want to mask
    :param mask_prob: the probability with which we hide entries in the dataframe
    :param feature_names: the names of the columns that can be hidden
    :param rng: the random generator object used to sample which rows have hidden entries
    :return: a tuple where the first entry is the training Numpy array with some possible Nan values,
             and the second entry is a numpy array with no Nan values and is the prices, and the
             third entry is the training Numpy array that has no masked values
    """
    res = []
    prices = train_df["realSum"]
    train_df = train_df.drop(columns=["realSum"])
    unmasked_train_df = train_df.copy(deep=True)

    train_df = train_df.drop(columns=["dist"]) # dropping distance to center

    column_indices = np.array([train_df.columns.get_loc(column_name) for column_name in feature_names])
    for i in range(len(train_df.index)):
        new_row = np.array(train_df.iloc[i])
        if rng.binomial(n=1, p=mask_prob) == 1:
            new_row[column_indices] = np.NaN
        res.append(new_row)
    
    assert np.sum(np.isnan(np.array(unmasked_train_df))) == 0
    return np.array(res), np.array(prices), unmasked_train_df

def create_test_arr(test_df)-> np.ndarray:
    """
    Converts the test dataframe to two numpy arrays (one with test features, other with test prices)
    :param test_df: the input dataframe
    :return: a tuple where the first entry is the test Numpy array with some possible Nan values
             and the second entry is a numpy array with no Nan values and is the prices
    """
    prices = test_df["realSum"]
    test_df = test_df.drop(columns=["realSum"])
    res = np.array(test_df)
    assert np.shape(test_df) == (len(test_df), len(test_df.columns))
    return res, np.array(prices)

def create_local_radius_centers_diags(
    masked_train_arr: np.ndarray,
    unmasked_train_df: pd.DataFrame,
    local_radius: float,
    rng: np.random.Generator,
    grid: bool = False
    ):
    """
    Creates grid used to define square robust uncertainty sets.
    """
    n, d = np.shape(masked_train_arr)
    # assert np.shape(unmasked_train_df) == (n, d)
    
    lng = unmasked_train_df["lng"].to_numpy()
    lat = unmasked_train_df["lat"].to_numpy()


    grid_center = find_city_center('london')
    d_mat = create_diag_matrix(grid_center)
    
    # scale longitude and latitude steps to be the same in kilometers
    lng_step = 2*local_radius / np.sqrt(d_mat[0, 0])
    lat_step = 2*local_radius / np.sqrt(d_mat[1, 1])

    # build grid (specified by botom left corner)
    grid_map = {}
    for i in range(n):
        lng_i, lat_i = lng[i], lat[i]
        lng_low_ = floor((lng_i - grid_center[0]) / lng_step)
        lat_low_ = floor((lat_i - grid_center[1]) / lat_step)

        grid_map[i] = (lng_low_, lat_low_)
    
    xs,ys = [],[] # build grid points for plotting
    for k,v in grid_map.items():
        xs.append(v[0]*lng_step + grid_center[0])
        xs.append(v[0]*lng_step + grid_center[0])
        xs.append((v[0]+1)*lng_step + grid_center[0])
        xs.append((v[0]+1)*lng_step + grid_center[0])
        
        ys.append(v[1]*lat_step + grid_center[1])
        ys.append((v[1]+1)*lat_step + grid_center[1])
        ys.append(v[1]*lat_step + grid_center[1])
        ys.append((v[1]+1)*lat_step + grid_center[1])

    if grid:
        # plotting stuff
        BBox = (lng.min(),   lng.max(), lat.min(), lat.max())

        #img = plt.imread('code/airbnb_exp/london_map.png')

        prop_lng = unmasked_train_df.lng[20]
        prop_lat = unmasked_train_df.lat[20]
        fig, ax = plt.subplots(figsize = (8,6))
        ax.scatter([prop_lng], [prop_lat+.007], zorder=1, alpha= 1, c='b', s=40, label="Property")
        ax.set_xlim(BBox[0],BBox[1])
        ax.set_ylim(BBox[2],BBox[3])
        #ax.imshow(img, zorder=0, extent = BBox, aspect= 'equal')

        # plot a horizontal line through each unique y coordinate in ys
        for y in np.unique(ys):
            ax.axhline(y, color='k', linestyle='-', alpha=0.2)
        
        # plot a vertical line through each unique x coordinate in xs
        for x in np.unique(xs):
            ax.axvline(x, color='k', linestyle='-', alpha=0.1)

        # plot the citycenter and label it
        city_center = find_city_center('london')

        ax.scatter(city_center[0], city_center[1], c='r', s=40,label="City Center")
        
        # given an ax object, plot a circle around the point (grid_center[0], grid_center[1]) going through the point
        # (prop_lng, prop_lat)
        # Calculate the radius of the circle
        radius = np.sqrt(((prop_lng - city_center[0]))**2 + ((prop_lat - city_center[1])**2))

        # Create the Circle patch object and add it to the axes
        from matplotlib.patches import Circle, Rectangle
        circle = Circle(city_center, radius, edgecolor='red', facecolor='none')
        # ax.add_patch(circle)

        # plot the square in the grid containing the property
        # first, round the property's lng and lat to the nearest grid center
        prop_grid_center = np.array([round((prop_lng - grid_center[0]) / lng_step - 1) * lng_step + grid_center[0],
                                        round((prop_lat - grid_center[1]) / lat_step) * lat_step + grid_center[1]])
        # plot the square
        ax.add_patch(Rectangle((prop_grid_center[0], prop_grid_center[1]), lng_step, lat_step,
                                edgecolor='green', facecolor='none'))
        
        ax.set_aspect(1.4)
        
        plt.legend()
        # save with tight layout
        plt.tight_layout()

        #plt.savefig("code/airbnb_exp/london_map_grid.pdf")
        return grid_map, grid_center, lng_step, lat_step


def create_diag_matrix(center: np.ndarray)->np.ndarray:
    """
    Creates the diagonal matrix that parameterizes the local distance around the center. Note that longitude is at coordinate
    index 0 and latitude is at coordinate index 1.
    
    :param center: a (2,) numpy array around which we locally approximate distance

    :return: the (2, 2) diagonal matrix
    """
    assert np.shape(center) == (2,)
    ellipsoid_matrix = np.zeros((2, 2))
    ellipsoid_matrix[0, 0] = (111.30 * np.cos(np.deg2rad(center[1]))) ** 2
    ellipsoid_matrix[1, 1] = 110.574 ** 2
    return ellipsoid_matrix

def find_city_center(city: str)->np.ndarray:
    """
    Returns the coordinates (longitude, latitude) of the city center of the provided city.
    :param city: a string with the city name
    :return: a numpy array of size (2,) which has the coordinates of the center of the corresponding city.
    """
    city_centers = {}
    city_centers["london"] = np.array([-0.127249, 51.507972])
    city_centers["berlin"] = np.array([13.413244, 52.521992])
    city_centers["lisbon"] = np.array([-9.140246, 38.712139])
    city_centers["paris"] = np.array([2.3524, 48.8565])
    assert city in city_centers
    return city_centers[city]