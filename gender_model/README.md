

*   **for Command-Line Arguments:** 
    ```bash
    python src/predict.py --names Priya Rahul Anjali  # Predict for these names
    python src/predict.py --input_file names.txt  # Read names from a file
    python src/predict.py --names Priya --model_path /path/to/your/model.pt --device cuda
    python src/predict.py --input_file names.txt --output_file predictions.csv --threshold 0.6
    ```
*   **Device Handling:** The `--device` argument allows you to specify "cpu" or "cuda".  It also includes a check to make sure CUDA is actually available before trying to use it.  This is *critical* for portability. The model is loaded directly onto the specified device using `map_location` in `torch.load`.
*   **Error Handling:**  Checks if the model file and input file (if provided) exist.  This prevents unexpected crashes.
*   **Flexible Input:**  You can provide names directly on the command line (`--names`) or from a text file (`--input_file`), one name per line.
*   **Optional Output File:**  The `--output_file` argument lets you save the predictions to a CSV file.
* **Threshold Parameter**: You can use --threshold for using different threshold for male classification.
*   **Clear Output:**  Prints a well-formatted table of results (or writes to a CSV).
*   **`load_model` Function:**  This function encapsulates the model loading logic, making the code cleaner.
*   **`predict_gender` Function:** This function encapsulates the prediction logic for a single name.
* **Docstrings**: Added docstrings to explain functions.
* **Main Block:** The `if __name__ == "__main__":` block ensures that the `main` function only runs when the script is executed directly (not when imported as a module).
* **File Handling** Uses `with open(...)` which automatically handles closing of files.

**How to Run (Examples)**

1.  **Predicting for a few names:**

    ```bash
    python src/predict.py --names Sameer Sakshi Rohan
    ```

2.  **Reading names from a file (`names.txt`):**

    Create a file named `names.txt` (in the same directory as `predict.py` or specify the full path) with names like this:

    ```
    Anika
    Rajesh
    Pooja
    David  # Example of a name that might not be in your training data
    ```

    Then run:

    ```bash
    python src/predict.py --input_file names.txt
    ```

3.  **Saving results to a CSV file:**

    ```bash
    python src/predict.py --input_file names.txt --output_file predictions.csv
    ```

4.  **Using a different threshold and specifying the device**
    ```bash
    python src/predict.py --names Sameer --model_path /home/sameer/sakshi/gender_model/models/indian_name_gender_model.pt --threshold 0.7 --device cuda
    ```
    If cuda device is not present it will automatically shift to CPU.

This improved and well-structured code provides a robust and user-friendly way to run your gender prediction model locally. It handles errors, uses command-line arguments, and is easily adaptable to different input methods and output formats. The use of separate files for the model and utility functions further enhances the organization and maintainability of the code. Remember to adjust the `--model_path` if your model file is located elsewhere.
