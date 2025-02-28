from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

# Évaluation du modèle sur l'ensemble de test
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = np.array(y_test)  # Convertir en array numpy

# Calculer l'accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Rapport de classification détaillé
print("\nRapport de classification:")
print(classification_report(y_true, y_pred, target_names=species_list))

# Matrice de confusion
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=species_list, yticklabels=species_list)
plt.title('Matrice de confusion')
plt.ylabel('Vrai label')
plt.xlabel('Prédiction')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# Calculer le Top-2 Accuracy
top2_accuracy = 0
for i, true_label in enumerate(y_true):
    pred_indices = np.argsort(predictions[i])[::-1][:2]  # Top 2
    if true_label in pred_indices:
        top2_accuracy += 1
top2_accuracy /= len(y_true)
print(f"Top-2 Accuracy: {top2_accuracy:.4f}")

# Analyse des erreurs
errors = y_true != y_pred
error_indices = np.where(errors)[0]

print(f"\nNombre d'erreurs: {len(error_indices)} sur {len(y_true)} images")
print("\nExemples d'erreurs de classification:")

for i in range(min(5, len(error_indices))):
    idx = error_indices[i]
    true_species = species_list[y_true[idx]]
    pred_species = species_list[y_pred[idx]]
    img_path = X_test[idx]
    print(f"Image: {os.path.basename(img_path)}")
    print(f"  Vrai: {true_species}, Prédit: {pred_species}")
    
    # Optimisation: identifier quelles espèces sont souvent confondues
    if i == 0:
        print("\nPaires d'espèces souvent confondues:")
        for i in range(len(species_list)):
            for j in range(i+1, len(species_list)):
                confusion = cm[i, j] + cm[j, i]
                if confusion > 2:  # Seuil arbitraire
                    print(f"  {species_list[i]} et {species_list[j]}: {confusion} erreurs")