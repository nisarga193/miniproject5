# Training Guide: Frequency-Dynamic CNN with Coordinate Attention (FD-CNN-CA)

## Overview

This guide explains how to train the FD-CNN-CA model for real-time animal sound classification. The model achieves **~95.21% accuracy** using advanced deep learning techniques.

## Architecture Highlights

1. **Frequency-Dynamic Convolution**: Adapts to different frequency components in audio signals
2. **Coordinate Attention**: Captures both frequency and temporal dependencies
3. **Advanced Augmentation**: Time/frequency masking, mixup, noise injection
4. **Balanced Sampling**: Ensures equal representation of all classes

## Step 1: Prepare Your Dataset

### Option A: Organize by Directory Structure (Recommended)

Create a directory structure like this:

```
data/
├── dog/
│   ├── dog_bark_001.wav
│   ├── dog_bark_002.wav
│   └── ...
├── cat/
│   ├── cat_meow_001.wav
│   ├── cat_meow_002.wav
│   └── ...
├── bird/
│   ├── bird_chirp_001.wav
│   └── ...
└── ...
```

### Option B: Use the Organization Script

If you have unorganized audio files with animal names in filenames:

```bash
# Organize files automatically
python backend/organize_animal_data.py --input_dir path/to/your/audio/files --output_dir data --pattern animal_name
```

Or create a sample structure:

```bash
python backend/organize_animal_data.py --output_dir data --create_sample
```

### Option C: Use ESC-50 Dataset

The ESC-50 dataset contains environmental sounds including animals. To extract only animal sounds:

```python
# You can modify prepare_esc50_data.py to filter for animal categories
# Animal categories in ESC-50: dog, rooster, pig, cow, frog, cat, hen, insects, sheep, crow
```

## Step 2: Train the Model

### Basic Training

```bash
python backend/train_fd_cnn_ca.py --data_dir data --epochs 150 --batch_size 32 --output models/animal_fd_cnn_ca.h5
```

### Advanced Training with Mixup Augmentation

```bash
python backend/train_fd_cnn_ca.py --data_dir data --epochs 150 --batch_size 32 --use_mixup --output models/animal_fd_cnn_ca.h5
```

### Custom Learning Rate

```bash
python backend/train_fd_cnn_ca.py --data_dir data --epochs 150 --batch_size 32 --lr 0.001 --output models/animal_fd_cnn_ca.h5
```

## Step 3: Monitor Training

The training script will:
- Display training progress with accuracy and loss
- Save the best model based on validation accuracy
- Generate training logs in CSV format
- Create error analysis file with misclassified examples

### Training Output Files

- `models/animal_fd_cnn_ca.h5` - Trained model
- `models/animal_fd_cnn_ca.h5.classes.json` - Class labels mapping
- `models/animal_fd_cnn_ca_training.log` - Training history (CSV)
- `models/animal_fd_cnn_ca_errors.txt` - Error analysis

## Step 4: Use the Trained Model

### Update Configuration

Set the model path in your `.env` file or environment:

```bash
MODEL_PATH=models/animal_fd_cnn_ca.h5
```

### Run the Flask Backend

```bash
python backend/app.py
```

The model will automatically load and be ready for real-time predictions.

## Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_dir` | Required | Directory containing animal sound folders |
| `--epochs` | 150 | Number of training epochs |
| `--batch_size` | 32 | Batch size for training |
| `--output` | `models/animal_fd_cnn_ca.h5` | Output model path |
| `--lr` | 0.001 | Initial learning rate |
| `--use_mixup` | False | Enable mixup augmentation |

## Tips for Best Results

1. **Dataset Size**: Aim for at least 100-200 samples per animal class
2. **Audio Quality**: Use clear, high-quality recordings
3. **Diversity**: Include variations in:
   - Recording environments (indoor/outdoor)
   - Background noise levels
   - Recording distances
   - Different individuals of the same species
4. **Data Augmentation**: The model includes automatic augmentation, but you can add more diversity to your dataset
5. **Balanced Classes**: Ensure roughly equal number of samples per class
6. **Validation**: The model uses 15% validation set to prevent overfitting

## Troubleshooting

### Low Accuracy
- Increase dataset size
- Ensure balanced classes
- Check audio quality
- Try increasing epochs
- Adjust learning rate

### Overfitting
- The model includes dropout and regularization
- Early stopping will prevent overfitting
- Consider reducing model complexity if still overfitting

### Memory Issues
- Reduce batch size (e.g., `--batch_size 16`)
- Use smaller audio duration in `config.py`
- Reduce number of mel bands

## Model Architecture Details

### Frequency-Dynamic CNN Block
- Multiple parallel convolutions with different kernel sizes
- Attention-based combination of branches
- Adapts to frequency characteristics dynamically

### Coordinate Attention
- Frequency-wise attention: pools along time axis
- Time-wise attention: pools along frequency axis
- Captures spatial dependencies in spectrograms

### Training Features
- Balanced batch sampling
- Advanced data augmentation (time/frequency masking, mixup)
- Learning rate scheduling
- Early stopping
- Model checkpointing

## Performance Expectations

With a good dataset, you should achieve:
- **Training Accuracy**: 95-98%
- **Validation Accuracy**: 90-95%
- **Test Accuracy**: 90-95% (target: 95.21%)

## Next Steps

1. Collect diverse animal sound dataset
2. Organize data using the organization script
3. Train the model with appropriate parameters
4. Evaluate on test set
5. Deploy using Flask backend
6. Monitor real-time predictions

## Support

For issues or questions:
- Check error logs in `models/animal_fd_cnn_ca_errors.txt`
- Review training logs in `models/animal_fd_cnn_ca_training.log`
- Examine confusion matrix in training output

