# Current state
Can I run the app in app.py? **Yes**, if you have the required python packages installed.
Can I re-train the model in dev.py? **Not with the original samples**, as I no longer have the image dataset. A new dataset in the same format would need to be created.
Is this learning project finished? **Yes**, as I do not intend to make any further changes or updates for now.

# Packages
1. Pytorch + torchvision for the machine learning part of the project.
2. PIL + imghdr for image processing.
3. customtkinter for GUI (more modern than original tkinter).
4. OS + pathlib for saving files and manipulating with directories.
5. PyInstaller for creating an executable file from app.py. _- just a test to see if everything works as an executable_

# Development
16.9.2023 - Basic model built, trained on the dataset containing 51 positive and 51 negative images.

17.9.2023 - Created a new folder of the dataset images without background, the model stopped overfitting at higher epoch counts, the average accuracy was improved. Created a mixed dataset containing all the images in two versions, with and without background.

![results](https://github.com/PopeCorn/myr/assets/117516270/63141241-2063-4a30-92dd-edf73ee4629e)

21.9.2023 - Changed the number of hidden units to 30. Coded out almost the entire GUI, app.py outputs the model's prediction in the terminal.

22.9.2023 - Increased the number of hidden units to 100. Added negative samples very similar to positive ones which reduced the number of false positives. Finished the GUI and added a short guide to it about the usage of the app.

23.9.2023 - Retrained the model on CPU with more samples, achieving 100% accuracy on test data once again.

17. and 20.3.2024 - Changed some variable names and removed obviously unnecessary features.

26.7.2024 - Returned to this project to fix my old code -> refined and created more functions, shortened lines, improved readability, stored results of training and testing to a .txt file.

![screenshot](https://github.com/user-attachments/assets/cb8b7b1c-1da3-481f-9e3e-119330b70247)

1.8.2024 - Last minor edits such as resolving missing imports and editing the README.
