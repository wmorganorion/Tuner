from data.process_data import get_filenames


train_filenames, test_filenames = get_filenames(data_path)

print(len(train_filenames), 'training data')
print(len(test_filenames), 'testing data')



def get_filenames(path):
    # if not os.path.exists(path):
        # os.makedirs(path)
        # download(path)

    image_paths = []
    for filename in os.listdir(path):
	    image_paths.append(os.path.join(path, filename))
			
		

    # 90% train images and 10% test images
    n_train_samples = int(len(image_paths) * 0.9)
    train_filenames = image_paths[:n_train_samples]
    test_filenames = image_paths[n_train_samples:]

    return train_filenames, test_filenames

