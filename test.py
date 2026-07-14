import pandas as pd
for f in ['turbojet_complete_dataset.csv','train.csv','test.csv','ground_truth.csv']:
    try:
        df = pd.read_csv(f'Dataset/{f}')
        print(f, df.shape, list(df.columns))
        print(df.head(3).to_string())
        print()
    except Exception as e:
        print(f, "ERROR:", e)