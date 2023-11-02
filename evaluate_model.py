import argparse
import os
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import sys

def evaluate_model(dataset_folder, model_path, output_file):
    try:
        if not os.path.exists(dataset_folder):
            raise FileNotFoundError(f"Error: Dataset folder '{dataset_folder}' not found.")

        # Check if the dataset folder is empty
        if not os.listdir(dataset_folder):
            raise ValueError(f"Error: Dataset folder '{dataset_folder}' is empty.")

        # Load the pre-trained model
        model = load_model(model_path)

        

        #Evaluate the model on the test data
        evaluation = model.evaluate()

        #Get the classification accuracy (or other relevant metric)
        accuracy = evaluation[1]  # Index 1 typically represents accuracy

        # Generate the architecture summary
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        model_summary = '\n'.join(model_summary)

        # Additional insights or observations
        insights = "I mesed up here but I tried my best."
        # Write the results to the output file
        with open(output_file, 'w') as file:
            file.write(f"Model Architecture Summary:\n{model_summary}\n\n")
            file.write(f"Classification Accuracy: {accuracy:.2%}\n\n")
            file.write(f"Additional Insights:\n{insights}")


        return accuracy
    except FileNotFoundError as e:
        print(e)
        return None
    except ValueError as e:
        print(e)
        return None

def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on a dataset.")
    parser.add_argument("dataset_folder", help="Path to the folder containing the evaluation dataset.")
    parser.add_argument("model_path", help="Path to the pre-trained model.")
    parser.add_argument("output_file", help="Path to the output file (e.g., 'output.txt').")

    args = parser.parse_args()
    dataset_folder = args.dataset_folder
    model_path = args.model_path
    output_file = args.output_file

    accuracy = evaluate_model(dataset_folder, model_path, output_file)
    
    if accuracy is not None:
        print(f"Classification Accuracy: {accuracy:.2%}")
        print(f"Results saved to '{output_file}'")
    
    # Exit the script with a clean exit code (0)
    sys.exit(0)

if __name__ == "__main__":
    main()
