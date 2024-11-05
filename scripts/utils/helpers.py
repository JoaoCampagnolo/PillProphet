import pandas as pd

def create_dataframe(data, fieldnames):
    df = pd.DataFrame(data)
    # Ensure all fieldnames are included in the DataFrame
    for field in fieldnames:
        if field not in df.columns:
            df[field] = None
    return df

def save_dataframe_to_csv(df, csv_file):
    df.to_csv(csv_file, index=False)
    print(f"Data has been saved to '{csv_file}'.")