# Mood Detection Using Songs

This project aims to detect the mood of songs using various deep learning models including VGG16, MobileNetV3, DenseNet121, EfficientNet-B0, and AlexNet. The dataset used for training and testing is the GTZAN dataset, which contains audio files of different music genres.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference](#inference)
- [Models](#models)
- [Results](#results)
- [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Vtsl-patel/moodDetectionUsingSongs.git
    cd moodDetectionUsingSongs
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Dataset Preparation

1. **Rename Files**: Rename the files in the dataset to include their genre names.
    ```python
    import os

    def rename_files(root_dir):
        root_dir = os.path.abspath(root_dir)
        for emotion in ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]:
            emotional_path = os.path.join(root_dir, str(emotion))
            list_emotions = os.listdir(emotional_path)
            for each_emotion in list_emotions:
                file_parts = each_emotion.split('.')
                piece = os.path.join(emotional_path, each_emotion)
                new_piece = os.path.join(emotional_path, emotion + str("_") + file_parts[1] + str(".") + file_parts[2])
                os.rename(piece, new_piece)
            print(f"Renamed {emotion} files")

    rename_files('path_to_dataset/genres_original')
    ```

2. **Create DataFrame**: Create a DataFrame with audio file paths and their corresponding labels.
    ```python
    def create_df(root_path):
        label = 0
        audio_name = []
        label_list = []
        for emotion in ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]:
            emotional_path = os.path.join(root_path, str(emotion))
            list_emotions = os.listdir(emotional_path)
            for each_emotion in list_emotions:
                piece = os.path.join(emotional_path, each_emotion)
                audio_name.append(piece)
                label_list.append(label)
            label += 1
        df = pd.DataFrame(columns = ['audio_name', 'emotion_label'])
        df["audio_name"] = audio_name
        df["emotion_label"] = label_list
        return df

    audio_df = create_df('path_to_dataset/genres_original')
    audio_df.to_csv('metaFile.csv', index = False)
    ```

### Training

1. **Load Data**: Prepare the data for training and testing.
    ```python
    from torch.utils.data import Dataset
    from torchvision import transforms

    class Data_Prepare(Dataset):
        def __init__(self, df, transform = None):
            self.df = df
            self.transform = transform
            self.hl = 512
            self.hi = 224
            self.wi = 224

        def __len__(self):
            return len(self.df)

        def __getitem__(self, index):
            if torch.is_tensor(index):
                index = index.tolist()
            audio_name = self.df.iloc[index, 0]
            try:
                audio_file, y = lib.load(audio_name)
            except Exception as e:
                print(f"Error loading audio file '{audio_name}': {e}")
                return None
            audio_window = audio_file[0:self.wi*self.hl]
            spectrogram = lib.feature.melspectrogram(y = audio_window, sr = y, n_mels = self.hi, fmax = 9000, hop_length = self.hl)
            plot = lib.power_to_db(spectrogram, ref=np.max)
            plot = cv2.resize(cv2.cvtColor(plot,cv2.COLOR_GRAY2RGB),(224,224))
            label = (self.df.iloc[index, -1])
            if self.transform:
                plot = self.transform(plot)
            return (plot, label)

    def data_preparation(Data_Class, Dataframe, Mean, Std, Batch_Size = 64, Shuffle = True):
        train_transform = transforms.Compose([
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(Mean, Std)
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(Mean, Std)
        ])

        train_dataset = Data_Class(df=Dataframe, transform=train_transform)
        test_dataset = Data_Class(df=Dataframe, transform=test_transform)
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(Mean, Std)])
        dataset_whole = Data_Class(df = Dataframe, transform = transform)
        test_split = 0.3
        random_seed= 42
        dataset_size = len(dataset_whole)
        indices = list(range(dataset_size))
        split = int(np.floor(test_split * dataset_size))
        if Shuffle==True:
          np.random.seed(random_seed)
          np.random.shuffle(indices)
        train_indices, test_indices = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)
        train_loader = torch.utils.data.DataLoader(dataset = dataset_whole, batch_size = Batch_Size, pin_memory = True, num_workers=0, sampler=train_sampler)
        test_loader = torch.utils.data.DataLoader(dataset = dataset_whole, batch_size = Batch_Size, pin_memory = True, num_workers=0, sampler=test_sampler)
        return train_loader, test_loader

    train, test = data_preparation(Data_Prepare, audio_df, Mean = [0, 0, 0], Std = [1, 1, 1], Batch_Size = 64, Shuffle = True)
    ```

2. **Model Training**: Train the model with the prepared data.
    ```python
    from torchvision import models
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    def model(Train_Loader, Test_Loader, Epochs, Model_Class = None, Loss_Function = None, Optimizer = None):
        scheduler = ReduceLROnPlateau(Optimizer, mode='min', patience=2, factor=0.1, verbose=True)
        outputs_train = []
        outputs_test = []
        y_true = []
        y_pred = []
        for Epoch in range(Epochs):
            running_loss_train = 0
            running_loss_test = 0
            correct_train = 0
            correct_test = 0
            for (image, label) in tqdm(Train_Loader):
                image = image.cuda()
                label = torch.tensor(label).cuda()
                out = Model_Class(image)
                loss, Optimizer = grad_change(Loss_Function = Loss_Function, Optimizer = Optimizer, Label = label, Predicted = out)
                running_loss_train += loss
                predicted_train = out.data.max(1, keepdim = True)[1]
                correct_train += predicted_train.eq(label.data.view_as(predicted_train)).sum()
                outputs_train.append((Epoch, running_loss_train / len(Train_Loader.dataset), 100 * correct_train / len(Train_Loader.dataset)))

            scheduler.step(running_loss_train)

            with torch.no_grad():
                for (image, label) in Test_Loader:
                    image = image.cuda()
                    label = torch.tensor(label).cuda()
                    out = Model_Class(image)
                    loss = Loss_Function(out, label)
                    running_loss_test += loss
                    predicted_test = out.data.max(1, keepdim = True)[1]
                    if Epoch == (Epochs - 1):
                        y_pred.extend(predicted_test.cpu().numpy())
                        y_true.extend(label.cpu().numpy())
                    correct_test += predicted_test.eq(label.data.view_as(predicted_test)).sum()
                    outputs_test.append((Epoch, running_loss_test / len(Test_Loader.dataset), 100 * correct_test / len(Test_Loader.dataset)))
        return Model_Class, outputs_train, outputs_test, y_pred, y_true
    ```

### Inference

1. **Load and Predict**: Load a saved model and predict the mood of a new song.
    ```python
    model_vgg16 = torch.load('path_to_saved_model/vgg16_model.pth')
    class_names = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    path_input = 'path_to_audio_file.wav'
    audio_file, y = lib.load(path_input)
    hl = 512
    hi = 224
    wi = 224
    audio_window = audio_file[0:wi*hl]
    spectrogram = lib.feature.melspectrogram(y = audio_window, sr = y, n_mels = hi, fmax = 9000, hop_length = hl)
    plot = lib.power_to_db(spectrogram, ref=np.max)
    plt.imshow(plot)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar()
    final_plot = cv2.resize(cv2.cvtColor(plot,cv2.COLOR_GRAY2RGB),(224,224))
    plot = torch.Tensor(final_plot).unsqueeze(0).cuda()
    model_vgg16.eval()
    out = model_vgg16(plot.permute(0,3,1,2).cuda())
    _, prediction = torch.max(out, 1)
    class_names[prediction]
    ```

### Models

1. **VGG16**: A variant of the VGGNet architecture with 16 layers.
2. **MobileNetV3-Small**: A smaller and more efficient version of MobileNetV3.
3. **DenseNet121**: A densely connected neural network with 121 layers.
4. **EfficientNet-B0**: A highly efficient model with compound scaling.
5. **AlexNet**: A classic deep learning model with 8 layers.

### Results

1. **Confusion Matrix**: A confusion matrix to evaluate model performance.
    ```python
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    conf_matrix = confusion_matrix(y_true, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt="d")
    ```

2. **Accuracy and Loss**: Plot training and testing accuracy and loss over epochs.

### License

This project is licensed under the nothing :).

