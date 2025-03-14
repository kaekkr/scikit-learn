import pandas as pd

from sklearn.datasets import fetch_file
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

url = "https://archive.ics.uci.edu/static/public/891/data.csv"
filepath = fetch_file(url)

df = pd.read_csv(filepath)

y = df["Diabetes_binary"].to_numpy()
x = df.drop(["ID", "Diabetes_binary"], axis=1).to_numpy()
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

x_train = x_train[0:100]
y_train = y_train[0:100]
x_test = x_test[8:9]

ml1 = KNeighborsClassifier(n_neighbors=5, algorithm="kd_tree").fit(x_train, y_train)
d1, n1 = ml1.kneighbors(x_test, return_distance=True)

auto_ml = KNeighborsClassifier(n_neighbors=5).fit(x_train, y_train)
auto_d, auto_n = auto_ml.kneighbors(x_test, return_distance=True)

ml2 = KNeighborsClassifier(n_neighbors=6, algorithm="kd_tree").fit(x_train, y_train)
d2, n2 = ml2.kneighbors(x_test, return_distance=True)

print(n1)
print(auto_n)
print(n2)

print(d1)
print(auto_d)
print(d2)
