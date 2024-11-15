import matplotlib.pyplot as plt
import pickle
import os
import math

# Verschiedene Listen von Modellkonfigurationen
# Jeder String repräsentiert eine bestimmte Konfiguration eines KI-Modells

twolayers_different_kernel_size = ['conv_16_32_ks_3_fc_128_lr_0.0003_opt_Adam', 'conv_16_32_ks_5_fc_128_lr_0.0003_opt_Adam','conv_32_128_ks_3_fc_256_lr_0.0003_opt_Adam', 
                       'conv_32_128_ks_5_fc_256_lr_0.0003_opt_Adam','conv_8_16_ks_3_fc_64_lr_0.0003_opt_Adam', 'conv_8_16_ks_5_fc_64_lr_0.0003_opt_Adam']

threelayers_opimizers = ['conv_16_32_64_ks_5_fc_128_64_lr_0.0003_opt_Adam', 'conv_16_32_64_ks_5_fc_128_64_lr_0.0003_opt_SGD', 'conv_24_64_128_ks_5_fc_256_128_lr_0.0003_opt_Adam',
                       'conv_24_64_128_ks_5_fc_256_128_lr_0.0003_opt_SGD','conv_12_24_48_ks_5_fc_96_48_lr_0.0003_opt_Adam','conv_12_24_48_ks_5_fc_96_48_lr_0.0003_opt_SGD']

batchsizes_learningrates = ['conv_24_64_128_ks_5_fc_256_128_lr_0.0003_opt_Adam_bs_32','conv_24_64_128_ks_5_fc_256_128_lr_0.0003_opt_Adam_bs_64','conv_24_64_128_ks_5_fc_256_128_lr_0.0003_opt_Adam_bs_128',
                       'conv_24_64_128_ks_5_fc_256_128_lr_0.0001_opt_Adam_bs_64', 'conv_24_64_128_ks_5_fc_256_128_lr_0.001_opt_Adam_bs_64']  

learning_rates = ['conv_24_64_128_ks_5_fc_256_128_lr_0.0002_opt_Adam_bs_64','conv_24_64_128_ks_5_fc_256_128_lr_0.0003_opt_Adam_bs_64', 'conv_24_64_128_ks_5_fc_256_128_lr_0.0004_opt_Adam_bs_64','conv_24_64_128_ks_5_fc_256_128_lr_0.0005_opt_Adam_bs_64',]

bigtest = ['conv_32_64_128_128_ks_5_fc_128_256_64_lr_0.0004_opt_Adam_bs_64', 'conv_24_64_128_ks_5_fc_256_128_lr_0.0004_opt_Adam_bs_64']

dropout = ['conv_32_64_128_ks_5_fc_256_128_lr_0.0004_opt_Adam_bs_64_do_0.5','conv_32_64_128_ks_5_fc_256_128_lr_0.0004_opt_Adam_bs_64_do_0.4','conv_32_64_128_ks_5_fc_256_128_lr_0.0004_opt_Adam_bs_64_do_0.3','conv_32_64_128_ks_5_fc_256_128_lr_0.0004_opt_Adam_bs_64_do_0.2','conv_32_64_128_ks_5_fc_256_128_lr_0.0004_opt_Adam_bs_64_do_0.1','conv_32_64_128_ks_5_fc_256_128_lr_0.0004_opt_Adam_bs_64_do_0']

end = ['conv_24_64_128_ks_5_fc_256_128_lr_0.0004_opt_Adam_bs_64', 'conv_32_64_128_ks_5_fc_256_128_lr_0.0004_opt_Adam_bs_64_do_0']
# Die Liste der Modellkonfigurationen, die wir visualisieren möchten
model_constructions = end

# Listen zum Speichern der Trainings- und Testgenauigkeiten für jedes Modell
train_accuracies = []
test_accuracies = []

# Für jede Modellkonfiguration
for model_construction in model_constructions:
    # Lade die Trainingsgenauigkeit aus einer Datei und füge sie zur Liste hinzu
    with open(os.path.join('models','train_accuracy', f'final_train_{model_construction}.pkl'), 'rb') as f:
        train_accuracies.append(pickle.load(f))

    # Lade die Testgenauigkeit aus einer Datei und füge sie zur Liste hinzu
    with open(os.path.join('models','test_accuracy', f'final_test_{model_construction}.pkl'), 'rb') as f:
        test_accuracies.append(pickle.load(f))

# Erstelle eine große Figur
plt.figure(figsize=(15, 9))

# Berechne die Anzahl der benötigten Zeilen für die Subplots
num_rows = math.ceil(len(model_constructions) / 5.0)

# Erstelle einen Subplot für alle Genauigkeiten
plt.subplot2grid((num_rows + 2, 5), (0, 0), colspan=5, rowspan=2)
mini = min([len(x) for x in test_accuracies])
for i, model_construction in enumerate(model_constructions):
    x_values = list(range(1, mini + 1))

    # Zeichne die Testgenauigkeiten für dieses Modell
    plt.plot(x_values, test_accuracies[i][:mini], label=f'Test Accuracy for {model_construction}')

plt.xlabel('Step')
plt.ylabel('Accuracy')
plt.legend()

# Erstelle einen Subplot für jedes Modell
for i, model_construction in enumerate(model_constructions):
    plt.subplot2grid((num_rows + 2, 5), (i // 5 + 2, i % 5))
    
    x_values = list(range(1, len(train_accuracies[i]) + 1))

    # Zeichne die Trainings- und Testgenauigkeiten für dieses Modell
    plt.plot(x_values, train_accuracies[i], label=f'Train Accuracy for {model_construction}')
    plt.plot(x_values, test_accuracies[i], label=f'Test Accuracy for {model_construction}')

    # Füge einen Titel zum Subplot hinzu
    plt.title(model_construction)

# Passe den Abstand zwischen den Subplots an
plt.subplots_adjust(wspace=0.5, hspace=0.5)

plt.tight_layout()

# Zeige die Figur an
plt.show()