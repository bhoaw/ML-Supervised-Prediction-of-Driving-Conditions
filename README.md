# ML-Supervised-Prediction-of-Driving-Conditions
The aim of this project was to predict driving style, road conditions or traffic state based on sensor inputs directly from a vehicle. 
Two data-sets were used, an Opel Corsa & a Peugeot 207.


<div class="cell code" execution_count="13" scrolled="true" tags="[]">

``` python
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
```

</div>

<div class="cell markdown">

## Data Exploration

</div>

<div class="cell code" execution_count="4">

``` python
#Download and concatenation of data
data11 = pd.read_csv('opel_corsa_01.csv', delimiter=';')
data12 = pd.read_csv('opel_corsa_02.csv', delimiter=';')
data21 = pd.read_csv('peugeot_207_01.csv', delimiter=';')
data22 = pd.read_csv('peugeot_207_02.csv', delimiter=';')
data   = pd.concat([data11, data12, data21, data22], ignore_index=True)

data
```
|       | AltitudeVariation | VehicleSpeedInstantaneous | VehicleSpeedAverage | VehicleSpeedVariance | VehicleSpeedVariation | LongitudinalAcceleration | EngineLoad | EngineCoolantTemperature | ManifoldAbsolutePressure | EngineRPM | MassAirFlow | IntakeAirTemperature | VerticalAcceleration | FuelConsumptionAverage |     roadSurface |                traffic |  drivingStyle |
|------:|------------------:|--------------------------:|--------------------:|---------------------:|----------------------:|-------------------------:|-----------:|-------------------------:|-------------------------:|----------:|------------:|---------------------:|---------------------:|-----------------------:|----------------:|-----------------------:|--------------:|
|   0   | NaN               | 0.000000                  | NaN                 | NaN                  | NaN                   | 0.0156                   | 25.490196  | 64.0                     | 100.0                    | 801.0     | 7.850000    | 22.0                 | -0.0078              | NaN                    | SmoothCondition | LowCongestionCondition | EvenPaceStyle |
|   1   | NaN               | 0.000000                  | NaN                 | NaN                  | NaN                   | 0.0156                   | 25.490196  | 64.0                     | 100.0                    | 803.0     | 7.890000    | 22.0                 | -0.0156              | NaN                    | SmoothCondition | LowCongestionCondition | EvenPaceStyle |
|   2   | NaN               | NaN                       | NaN                 | NaN                  | NaN                   | 0.0273                   | 25.882353  | 64.0                     | 100.0                    | 800.0     | 7.770000    | 22.0                 | -0.0273              | NaN                    | SmoothCondition | LowCongestionCondition | EvenPaceStyle |
|   3   | NaN               | NaN                       | NaN                 | NaN                  | NaN                   | 0.0391                   | 25.882353  | 64.0                     | 100.0                    | 798.0     | 7.770000    | 22.0                 | -0.0273              | NaN                    | SmoothCondition | LowCongestionCondition | EvenPaceStyle |
|   4   | NaN               | 0.000000                  | NaN                 | NaN                  | NaN                   | 0.0469                   | 25.882353  | 65.0                     | 100.0                    | 798.0     | 7.940000    | 22.0                 | -0.0312              | NaN                    | SmoothCondition | LowCongestionCondition | EvenPaceStyle |
|  ...  | ...               | ...                       | ...                 | ...                  | ...                   | ...                      | ...        | ...                      | ...                      | ...       | ...         | ...                  | ...                  | ...                    | ...             | ...                    | ...           |
| 24952 | 1.000000          | 28.799999                 | 28.559999           | 57.190571            | 3.600000              | -0.0292                  | 25.882353  | 81.0                     | 115.0                    | 1755.5    | 20.469999   | 25.0                 | -0.1661              | 14.578003              | SmoothCondition | LowCongestionCondition | EvenPaceStyle |
| 24953 | 1.699997          | 30.599998                 | 28.529999           | 57.010266            | 1.799999              | -0.0304                  | 11.764706  | 81.0                     | 106.0                    | 736.5     | 17.740000   | 25.0                 | -0.1987              | 14.585642              | SmoothCondition | LowCongestionCondition | EvenPaceStyle |
| 24954 | 1.800003          | 29.699999                 | 28.499999           | 56.883045            | -0.900000             | -0.1684                  | 98.039215  | 81.0                     | 106.0                    | 1254.0    | 9.520000    | 24.0                 | -0.1156              | 14.547294              | SmoothCondition | LowCongestionCondition | EvenPaceStyle |
| 24955 | 2.100006          | 29.699999                 | 28.409999           | 56.160910            | 0.000000              | -0.0644                  | 79.607841  | 80.0                     | 112.0                    | 1254.0    | 14.910000   | 23.0                 | -0.0760              | 14.546828              | SmoothCondition | LowCongestionCondition | EvenPaceStyle |
| 24956 | 1.500000          | 33.299999                 | 28.349999           | 55.340843            | 3.600000              | -0.1817                  | 80.000000  | 80.0                     | 113.0                    | 1363.5    | 15.330000   | 23.0                 | -0.0605              | 14.554068              | SmoothCondition | LowCongestionCondition | EvenPaceStyle |

</div>

<div class="cell code" execution_count="5">

``` python
counts_rs = data['roadSurface'].value_counts(sort=False)

total_rs = 0
for entry in counts_rs:
    total_rs += entry

for i in range(len(counts_rs)):
    print(counts_rs.index[i] , ':' , round((counts_rs[i]/total_rs)*100,2), '%' )
```

<div class="output stream stdout">

    SmoothCondition : 61.07 %
    UnevenCondition : 25.91 %
    FullOfHolesCondition : 13.02 %

</div>

</div>

<div class="cell code" execution_count="6">

``` python
counts_t = data['traffic'].value_counts(sort=False)

total_t = 0
for entry in counts_t:
    total_t += entry

for i in range(len(counts_t)):
    print(counts_t.index[i] , ':' , round((counts_t[i]/total_t)*100,2), '%' )
```

<div class="output stream stdout">

    LowCongestionCondition : 75.21 %
    NormalCongestionCondition : 12.71 %
    HighCongestionCondition : 12.09 %

</div>

</div>

<div class="cell code" execution_count="7">

``` python
counts_ds = data['drivingStyle'].value_counts(sort=False)

total_ds = 0
for entry in counts_ds:
    total_ds += entry

for i in range(len(counts_ds)):
    print(counts_ds.index[i] , ':' , round((counts_ds[i]/total_ds)*100,2), '%' )
```

<div class="output stream stdout">

    EvenPaceStyle : 88.51 %
    AggressiveStyle : 11.49 %

</div>

</div>

<div class="cell markdown">

## Data Prep

</div>

<div class="cell code" execution_count="8">

``` python
#Input data
x = data.values[::, 0:14]

#NaN, Standartization, Label Encoder
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy="most_frequent")
x = imp.fit_transform(x)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(x)
x = scaler.fit_transform(x)
```

</div>

<div class="cell code" execution_count="9">

``` python
def processYandSplit(range):
    #Feature Assignment
    y = data.values[::, range]

    enc = OrdinalEncoder()
    # enc = OneHotEncoder()
    y = enc.fit_transform(y)

    #Split into training & test data with a ratio of 80:20
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Make y_train 1 dimensional
    y_train = y_train.ravel()
    return x_train, x_test, y_train, y_test
```

</div>

<div class="cell markdown">

# Models

</div>

<div class="cell code" execution_count="10">

``` python
def computeSVC(x_train, x_test, y_train, y_test):
    #Train the SVC model
    from sklearn.svm import SVC
    svc = SVC(kernel='linear')
    model = svc.fit(x_train, y_train)

    #Predict the test data
    resultSVC = svc.predict(x_test)

    # Assess the success with a confusion matrix
    conf = confusion_matrix(y_test, resultSVC)
    pd.DataFrame(conf, index=svc.classes_, columns=svc.classes_)
    
    return resultSVC, conf
```

</div>

<div class="cell code" execution_count="11" scrolled="true">

``` python
def computeLR(x_train, x_test, y_train, y_test, solver='lbfgs', penalty='l2', C=0.1):
    from sklearn.linear_model import LogisticRegression
    #Train the LR model
    clf = LogisticRegression(random_state=0, solver=solver, penalty=penalty, C=C, multi_class='multinomial')

    model2 = clf.fit(x_train, y_train)
    #Predict the test data
    resultLR = clf.predict(x_test)

    # Assess the success with a confusion matrix
    conf = confusion_matrix(y_test, resultLR)
    pd.DataFrame(conf, index=clf.classes_, columns=clf.classes_)
    
    return resultLR, conf
```

</div>

<div class="cell markdown">

# Results

</div>

<div class="cell code" execution_count="24">

``` python
results_df = pd.DataFrame({})
# For each target feature:
for i in [14, 15, 16]:
    # Split data sets
    x_train, x_test, y_train, y_test = processYandSplit(slice(i,i+1))
    # Compute SVC
    resultSVC, confSVC = computeSVC(x_train, x_test, y_train, y_test)
    # Compute Logistic Regression 
    resultLR, confLR   = computeLR(x_train, x_test, y_train, y_test)
    # Build a results DataFrame
    accuracy_data = [[accuracy_score(y_test, resultSVC), balanced_accuracy_score(y_test, resultSVC), f1_score(y_test, resultSVC, average='macro'), accuracy_score(y_test, resultLR), balanced_accuracy_score(y_test, resultLR), f1_score(y_test, resultLR, average='macro')]]
    feature_results =  pd.DataFrame(accuracy_data, index = [data.columns[i]], columns=["SVC Acc", "SVC Bal Acc", "SVC F1 Mean", "LR Acc", "LR Bal Acc", "LR F1 Mean"])
    results_df = pd.concat([results_df, feature_results])
```

</div>

<div class="cell code" execution_count="25">

``` python
results_df
```
|              |  SVC Acc | SVC Bal Acc | SVC F1 Mean |   LR Acc | LR Bal Acc | LR F1 Mean |
|-------------:|---------:|------------:|------------:|---------:|-----------:|-----------:|
|  roadSurface | 0.782652 | 0.737606    | 0.729651    | 0.778446 | 0.715456   | 0.719583   |
|    traffic   | 0.775441 | 0.468239    | 0.455450    | 0.782051 | 0.487309   | 0.499387   |
| drivingStyle | 0.881611 | 0.500000    | 0.468540    | 0.880409 | 0.512502   | 0.496514   |
