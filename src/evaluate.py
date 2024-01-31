import matplotlib.pyplot as plt
import pickle
import os
import math

model_constructions = ['16_32-8-8_128', '16_32-8-8_128', '16_32-8-8_128', '16_32-8-8_128', '16_32-8-8_128', '16_32-8-8_128']  # replace with your list of model constructions

train_accuracies = []
test_accuracies = []

for model_construction in model_constructions:
    with open(os.path.join('models','train_accuracy', f'final_train_{model_construction}.pkl'), 'rb') as f:
        train_accuracies.append(pickle.load(f))

    with open(os.path.join('models','test_accuracy', f'final_test_{model_construction}.pkl'), 'rb') as f:
        test_accuracies.append(pickle.load(f))

# Create a large figure with a 2:3 width:height ratio
plt.figure(figsize=(15, 8))

# Calculate the number of rows needed for the subplots
num_rows = math.ceil(len(model_constructions) / 5.0)

# Create a subplot for all accuracies, taking up two rows at the top of the figure
plt.subplot2grid((num_rows + 2, 5), (0, 0), colspan=5, rowspan=2)

for i, model_construction in enumerate(model_constructions):
    # Generate x values. Assuming they should go from 1 to the length of the accuracy list + 1
    x_values = list(range(1, len(train_accuracies[i]) + 1))

    # Plot test accuracies for this model
    plt.plot(x_values, test_accuracies[i], label=f'Test Accuracy for {model_construction}')

plt.xlabel('Step')
plt.ylabel('Accuracy')
plt.legend()

# Create a subplot for each model
for i, model_construction in enumerate(model_constructions):
    plt.subplot2grid((num_rows + 2, 5), (i // 5 + 2, i % 5))
    
    x_values = list(range(1, len(train_accuracies[i]) + 1))
    
    plt.plot(x_values, train_accuracies[i], label=f'Train Accuracy for {model_construction}')
    plt.plot(x_values, test_accuracies[i], label=f'Test Accuracy for {model_construction}')

    # Add a title to the subplot
    plt.title(model_construction)

# Adjust the space between the subplots
plt.subplots_adjust(wspace=0.5, hspace=0.5)

plt.tight_layout()
plt.show()