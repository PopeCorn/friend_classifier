import os

if not os.path.exists('!RESULTS/'):
    os.mkdir('!RESULTS/')

with open('example.txt', 'w') as file:
    file.write('your results are positive :D')
