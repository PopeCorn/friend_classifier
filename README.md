# myr
#mýr is a machine learning project for classifying whether Mojmýr is in an image or not.

# The process
16.9.2023 - Basic model built, trained on the dataset containing 51 positive and 51 negative images.

17.9.2023 - Created a new folder of the #mýr dataset images, this time without background, to see if the training process will be affected; got better results, the model stopped overfitting at higher epoch counts, the average accuracy was improved. Then, created a mixed dataset containing all the images in two versions, with and without background, to achieve better versatility of the model for real-life use.
![results](https://github.com/PopeCorn/myr/assets/117516270/63141241-2063-4a30-92dd-edf73ee4629e)

# A little bit extra info
I got the idea to make this project at the start of summer of 2023, which I spend learning PyTorch, a library I wanted to be my tool for creating this project. After getting to section 4 of the excellent Mr. Bourke's course, I started working on #mýr, experimenting with Siamese networks that I hoped would be the best solution for my problem. Unfortunately, I ran into a whole bunch of errors and because I didn't even fully know what I was doing, this attempt ended unsuccesfully. Then, I decided to go back to what I know and built a model that replicates the TinyVGG CNN structure - and this finally worked. And that is where I'm at right now. 
