This is a sklearn compatible Nan imputer based on PCA and LinearRegression.
Performs inline with KNNImputer (Better of worst depending on feature type) but about 30x faster.

Check the examples/nb.ipynb for usage and benchmark comparison. 

```
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from pca_imputer import PCAImputer

X, y = datasets.fetch_california_housing(return_X_y=True, as_frame=True)
np.random.seed(42)
nans = np.random.binomial(1, 0.05, size = X.shape,) == 1
X_nan = X.where(~nans, np.nan)

imputer_pca = PCAImputer(n_components=5)
X_pca_filled = pd.DataFrame(imputer_pca.fit_transform(X_nan), columns=X_nan.columns, index=X_nan.index)
r2_score(X, X_pca_filled, multioutput='uniform_average') # 97%
```
