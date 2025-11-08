"""
Helper script to organize animal sound data for training.
Expects audio files with animal names in filenames or organizes by directory structure.
"""
import os
import shutil
import argparse
from pathlib import Path


def organize_by_directory(input_dir, output_dir, pattern='animal_name'):
    """
    Organize audio files into directories by animal name.
    
    Args:
        input_dir: Directory containing audio files (can have subdirectories)
        output_dir: Output directory where organized folders will be created
        pattern: How to extract animal names ('animal_name' or 'directory')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    animal_classes = set()
    file_mapping = {}
    
    print(f'Scanning {input_dir} for audio files...')
    
    # Find all audio files
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if Path(file).suffix.lower() in audio_extensions:
                file_path = os.path.join(root, file)
                
                if pattern == 'animal_name':
                    # Extract animal name from filename (e.g., "dog_bark_001.wav" -> "dog")
                    # Try different patterns
                    name_parts = Path(file).stem.lower().split('_')
                    if len(name_parts) > 0:
                        animal_name = name_parts[0]  # First part as animal name
                    else:
                        animal_name = Path(file).stem.lower()
                else:
                    # Use directory name as animal class
                    animal_name = os.path.basename(root).lower()
                
                # Clean up animal name
                animal_name = animal_name.strip().replace(' ', '_')
                
                if animal_name not in animal_classes:
                    animal_classes.add(animal_name)
                    file_mapping[animal_name] = []
                
                file_mapping[animal_name].append(file_path)
    
    print(f'\nFound {len(animal_classes)} animal classes:')
    for animal in sorted(animal_classes):
        print(f'  {animal}: {len(file_mapping[animal])} files')
    
    # Create directories and copy files
    print(f'\nOrganizing files into {output_dir}...')
    for animal, files in file_mapping.items():
        animal_dir = os.path.join(output_dir, animal)
        os.makedirs(animal_dir, exist_ok=True)
        
        for idx, src_file in enumerate(files):
            ext = Path(src_file).suffix
            dst_file = os.path.join(animal_dir, f'{animal}_{idx+1:04d}{ext}')
            shutil.copy2(src_file, dst_file)
        
        print(f'  Copied {len(files)} files to {animal}/')
    
    print(f'\n✅ Organization complete!')
    print(f'Data is organized in: {output_dir}')
    print(f'\nYou can now train the model with:')
    print(f'  python backend/train_fd_cnn_ca.py --data_dir {output_dir}')


def create_sample_structure(output_dir):
    """Create a sample directory structure for animal sounds"""
    sample_animals = ['dog', 'cat', 'bird', 'cow', 'pig', 'chicken', 'horse', 'sheep']
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f'Creating sample directory structure in {output_dir}...')
    for animal in sample_animals:
        animal_dir = os.path.join(output_dir, animal)
        os.makedirs(animal_dir, exist_ok=True)
        print(f'  Created directory: {animal}/')
    
    print(f'\n✅ Sample structure created!')
    print(f'Please add audio files (.wav, .mp3, etc.) to each animal directory.')
    print(f'Example: {output_dir}/dog/dog_bark_001.wav')


def main():
    parser = argparse.ArgumentParser(
        description='Organize animal sound data for training'
    )
    parser.add_argument(
        '--input_dir', 
        help='Input directory containing audio files'
    )
    parser.add_argument(
        '--output_dir', 
        required=True,
        help='Output directory for organized data'
    )
    parser.add_argument(
        '--pattern',
        choices=['animal_name', 'directory'],
        default='animal_name',
        help='How to extract animal names from files'
    )
    parser.add_argument(
        '--create_sample',
        action='store_true',
        help='Create a sample directory structure instead'
    )
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_structure(args.output_dir)
    elif args.input_dir:
        organize_by_directory(args.input_dir, args.output_dir, args.pattern)
    else:
        parser.print_help()
        print('\nError: Either --input_dir or --create_sample must be specified')


if __name__ == '__main__':
    main()

