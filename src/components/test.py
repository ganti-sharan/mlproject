import pandas as pd

raw_data = pd.read_csv('/Users/sharan/Documents/ML Study/MLProject/artifacts/raw.csv')
numerical_columns = raw_data.select_dtypes(exclude="object").columns.tolist()
cat_columns = raw_data.select_dtypes(include="object").columns.tolist()

