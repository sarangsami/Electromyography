import os
import pandas as pd

# Define the list of persons and movements
Persons = ['EMG-S1', 'EMG-S2', 'EMG-S4', 'EMG-S5',
           'EMG-S6', 'EMG-S8', 'EMG-S9', 'EMG-S10']
Movements = ['HC', 'II', 'LL', 'MM', 'RR', 'TI', 'TL', 'TM', 'TR', 'TT']

# Specify the base directory
base_directory = "output"
output_directory = os.path.join(base_directory, 'allPersons')

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Loop through each movement
for movement in Movements:
    all_data_c2 = []
    all_data_d1 = []
    all_data_d2 = []

    # Loop through each person
    for person in Persons:
        folder_path = os.path.join(base_directory, person, movement)

        # Loop through each file type (c2, d1, d2)
        for file_type in ['c2', 'd1', 'd2']:
            file_path = os.path.join(
                folder_path, f'all_{movement}_{file_type}_features.csv')

            # Check if the file exists
            if os.path.exists(file_path):
                # Read the CSV file into a DataFrame
                df = pd.read_csv(file_path)

                # Append the DataFrame to the corresponding list
                if file_type == 'c2':
                    all_data_c2.append(df)
                elif file_type == 'd1':
                    all_data_d1.append(df)
                elif file_type == 'd2':
                    all_data_d2.append(df)

            else:
                print(f"File not found: {file_path}")

    # Concatenate all DataFrames for the current movement and file type
    if all_data_c2:
        combined_df_c2 = pd.concat(all_data_c2, ignore_index=True)
        output_path_c2 = os.path.join(
            output_directory, movement, f'{movement}_c2_features.csv')
        os.makedirs(os.path.dirname(output_path_c2), exist_ok=True)
        combined_df_c2.to_csv(output_path_c2, index=False)
        print(f"Combined data for {movement} and c2 saved to {output_path_c2}")

    if all_data_d1:
        combined_df_d1 = pd.concat(all_data_d1, ignore_index=True)
        output_path_d1 = os.path.join(
            output_directory, movement, f'{movement}_d1_features.csv')
        os.makedirs(os.path.dirname(output_path_d1), exist_ok=True)
        combined_df_d1.to_csv(output_path_d1, index=False)
        print(f"Combined data for {movement} and d1 saved to {output_path_d1}")

    if all_data_d2:
        combined_df_d2 = pd.concat(all_data_d2, ignore_index=True)
        output_path_d2 = os.path.join(
            output_directory, movement, f'{movement}_d2_features.csv')
        os.makedirs(os.path.dirname(output_path_d2), exist_ok=True)
        combined_df_d2.to_csv(output_path_d2, index=False)
        print(f"Combined data for {movement} and d2 saved to {output_path_d2}")


# NORMALIZE
normalized_directory = "allPersonsNormalized"
output_normalized_directory = os.path.join(
    base_directory, normalized_directory)

# Create the output directory if it doesn't exist
os.makedirs(output_normalized_directory, exist_ok=True)

# Loop through each movement
for movement in Movements:
    for file_type in ['c2', 'd1', 'd2']:
        # Construct the file path for the concatenated CSV file
        input_file_path = os.path.join(
            output_directory, movement, f'{movement}_{file_type}_features.csv')

        # Check if the input file exists
        if os.path.exists(input_file_path):
            # Read the CSV file into a DataFrame
            concatenated_df = pd.read_csv(input_file_path)

            # Apply the normalization formula to each value in each column
            for column in concatenated_df.columns:
                max_value = concatenated_df[column].max()
                min_value = concatenated_df[column].min()
                print(f"Column: {column}, Max: {max_value}, Min: {min_value}")

                # Apply the normalization formula to the entire column
                concatenated_df[column] = (
                    concatenated_df[column] - min_value) / (max_value - min_value)
              

            # Construct the file path for the normalized CSV file
            output_file_path = os.path.join(
                output_normalized_directory, movement, f'{movement}_{file_type}_normalized_features.csv')

            # Create the output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

            # Save the normalized DataFrame to CSV
            concatenated_df.to_csv(output_file_path, index=False)
            print(
                f"Normalized data for {movement}_{file_type} saved to {output_file_path}")

            print("\n" + "="*40 + "\n")  # Separate output for readability
        else:
            print(f"Input file not found: {input_file_path}")


# Combine all C2, D1, and D2 movements
all_combined_c2 = pd.concat([pd.read_csv(os.path.join(output_normalized_directory, movement, f'{movement}_c2_normalized_features.csv'))
                            for movement in Movements], ignore_index=True)

all_combined_d1 = pd.concat([pd.read_csv(os.path.join(output_normalized_directory, movement, f'{movement}_d1_normalized_features.csv'))
                            for movement in Movements], ignore_index=True)

all_combined_d2 = pd.concat([pd.read_csv(os.path.join(output_normalized_directory, movement, f'{movement}_d2_normalized_features.csv'))
                            for movement in Movements], ignore_index=True)

# Save the combined DataFrames to CSV
output_combined_directory = os.path.join(
    base_directory, 'allMovementsCombined')
os.makedirs(output_combined_directory, exist_ok=True)

all_combined_c2.to_csv(os.path.join(
    output_combined_directory, 'all_movements_c2_combined.csv'), index=False)
all_combined_d1.to_csv(os.path.join(
    output_combined_directory, 'all_movements_d1_combined.csv'), index=False)
all_combined_d2.to_csv(os.path.join(
    output_combined_directory, 'all_movements_d2_combined.csv'), index=False)

print("Combined data for all movements and C2, D1, D2 saved.")

# Initialize empty arrays for labels
c2_labels = []
d1_labels = []
d2_labels = []

# Loop through each movement
for movement in Movements:
    # Read the normalized CSV file for each movement and file type
    c2_df = pd.read_csv(os.path.join(output_normalized_directory, movement, f'{movement}_c2_normalized_features.csv'))
    d1_df = pd.read_csv(os.path.join(output_normalized_directory, movement, f'{movement}_d1_normalized_features.csv'))
    d2_df = pd.read_csv(os.path.join(output_normalized_directory, movement, f'{movement}_d2_normalized_features.csv'))

    # Assign labels based on the number of occurrences for each movement
    c2_labels += [Movements.index(movement)] * len(c2_df)
    d1_labels += [Movements.index(movement)] * len(d1_df)
    d2_labels += [Movements.index(movement)] * len(d2_df)

# Convert the lists to numpy arrays for further use, if needed
import numpy as np
c2_labels = np.array(c2_labels)
d1_labels = np.array(d1_labels)
d2_labels = np.array(d2_labels)

# Save labels to CSV
labels_directory = os.path.join(base_directory, 'allMovementsCombined', 'labels')
os.makedirs(labels_directory, exist_ok=True)

c2_labels_df = pd.DataFrame({'c2_labels': c2_labels})
d1_labels_df = pd.DataFrame({'d1_labels': d1_labels})
d2_labels_df = pd.DataFrame({'d2_labels': d2_labels})

c2_labels_path = os.path.join(labels_directory, 'c2_labels.csv')
d1_labels_path = os.path.join(labels_directory, 'd1_labels.csv')
d2_labels_path = os.path.join(labels_directory, 'd2_labels.csv')

c2_labels_df.to_csv(c2_labels_path, index=False)
d1_labels_df.to_csv(d1_labels_path, index=False)
d2_labels_df.to_csv(d2_labels_path, index=False)
