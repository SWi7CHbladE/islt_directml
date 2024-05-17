import os
import pickle
import gzip

def combine_pickles(input_paths, output_path):
    combined_data = []

    for path in input_paths:
        with open(path, 'rb') as f:
            with gzip.GzipFile(fileobj=f) as gz:
                data = pickle.load(gz)
                combined_data.extend(data)
                print(f"Loaded {len(data)} items from {path}")

    with open(output_path, 'wb') as f:
        with gzip.GzipFile(fileobj=f, mode='wb') as gz:
            pickle.dump(combined_data, gz)
            print(f"Combined data saved to {output_path}")

if __name__ == "__main__":
    # List of input pickle files
    input_paths = [
        "Dataset/Pickles/excel_data0.train",
        "Dataset/Pickles/excel_data1.train",
        "Dataset/Pickles/excel_data2.train",
        "Dataset/Pickles/excel_data3.train",
        "Dataset/Pickles/excel_data4.train",
        "Dataset/Pickles/excel_data5.train",
        "Dataset/Pickles/excel_data6.train",
        "Dataset/Pickles/excel_data7.train",
        "Dataset/Pickles/excel_data8.train",
        "Dataset/Pickles/excel_data9.train",
        # Add more paths as needed
    ]
    

    # Output path for the combined pickle file
    output_path = "Dataset/Pickles/excel_data.train"

    combine_pickles(input_paths, output_path)
