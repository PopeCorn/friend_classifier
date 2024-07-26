# The Development

16.9.2023 - Basic model built, trained on the dataset containing 51 positive and 51 negative images.

17.9.2023 - Created a new folder of the #mýr dataset images, this time without background, got better results, the model stopped overfitting at higher epoch counts, the average accuracy was improved. Created a mixed dataset containing all the images in two versions, with and without background.

![results](https://github.com/PopeCorn/myr/assets/117516270/63141241-2063-4a30-92dd-edf73ee4629e)

21.9.2023 - Changed the number of hidden units to 30. Coded out almost the entire GUI, app.py outputs the model's prediction in the terminal.

22.9.2023 - Increased the number of hidden units to 100. Added negative samples very similar to positive ones which reduced the number of false positives. Finished the GUI and added a short guide to it about the usage of the app.

23.9.2023 - Retrained the model on CPU with more samples, achieving 100% accuracy on test data once again.

17. and 20.3.2024 - Changed some variable names and removed obviously unnecessary features.

26.7.2024 - Returned to this project to fix my old, ugly code -> refined and created more functions, shortened lines, improved readability, stored results of training and testing to a .txt file.

![screenshot](https://github.com/user-attachments/assets/cb8b7b1c-1da3-481f-9e3e-119330b70247)
