#Trainer
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load data
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

# Inspect the data to check for consistency
data_list = data_dict['data']
print(f"First element length: {len(data_list[0])}")
print(f"Data length: {len(data_list)}")

# Find the maximum length of the data entries
max_length = max(len(item) for item in data_list)
print(f"Maximum length of data entries: {max_length}")

# Fix inhomogeneous part by padding or truncating data
fixed_data = [np.pad(item, (0, max_length - len(item)), 'constant') if len(item) < max_length else item[:max_length] for item in data_list]

# Convert to numpy array
data = np.array(fixed_data)
labels = np.asarray(data_dict['labels'])

# Verify the shape
print(f"Data shape: {data.shape}")
print(f"Labels shape: {labels.shape}")

# Split data
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions
y_predict = model.predict(x_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_predict)
precision = precision_score(y_test, y_predict, average='weighted')
recall = recall_score(y_test, y_predict, average='weighted')
f1 = f1_score(y_test, y_predict, average='weighted')

print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Precision: {precision * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')
print(f'F1 Score: {f1 * 100:.2f}%')

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model':model},f)
