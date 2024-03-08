# Visualize the data
plt.figure(figsize=(10, 10))
for images, labels in validation_dataset.take(1):  # labels[0,1,0]
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[np.argmax(labels[i])])
        plt.axis("off")
plt.show()
plt.close("off")