import numpy as np
import tensorflow as tf
from preprocess.pamap2.data_loader import get_pamap2_data

# Helper function to print and write to file simultaneously


def print_and_write(f, *args):
    print(*args)
    f.write(" ".join(str(a) for a in args) + "\n")


# Open the output file
output_file = "random_input.txt"
with open(output_file, "w") as f:

    # Step 1: Load PAMAP2 dataset (where the transform will take place)
    (train_x, train_y), (val_x, val_y), (test_x,
                                         test_y), raw_test_y = get_pamap2_data(verbose=True)

    # Step 2: Pick a random test sample
    idx = np.random.randint(0, test_x.shape[0])
    sample_x = test_x[idx:idx+1]
    sample_y = test_y[idx]

    print_and_write(f, f"\nRandom sample index: {idx}")
    print_and_write(
        f, "Sample input shape (window_size x features):", sample_x.shape)
    print_and_write(f, "Sample true label (one-hot):", sample_y)
    print_and_write(f, "Sample true label (class index):", np.argmax(sample_y))

    # Step 3: Load pretrained model
    model = tf.keras.models.load_model(
        'saved_model/pamap2')  # adjust path if needed
    print_and_write(f, "Model input shape:", model.input_shape)

    # Step 4: Run inference (where we're fitting the data)
    pred = model.predict(sample_x)
    pred_class = np.argmax(pred, axis=-1)
    print_and_write(f, "Predicted class index:", pred_class[0])
    print_and_write(f, "Predicted class probabilities:", pred[0])

    # Step 5: Show raw segment data (first 5 timesteps)
    print_and_write(f, "\nRaw segment data (first 5 timesteps):")
    for row in sample_x[0][:5, :]:
        print_and_write(f, row)

print(f"\nAll sample information saved to {output_file}")


# Creating the embedding model
embedding_model = tf.keras.Model(
    inputs=model.input,
    outputs=model.get_layer("dense_12").output
)

# Extracting the embedding for this one sample
embedding_vec = embedding_model.predict(sample_x)

np.save("single_embedding.npy", embedding_vec)
print_and_write(f, "Embedding shape:", embedding_vec.shape)
