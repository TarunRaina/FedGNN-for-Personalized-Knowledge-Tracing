import pandas as pd

# Remove the display row and column caps
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

df = pd.read_csv("junyi_ProblemLog_original.csv", nrows=100)
print(df)