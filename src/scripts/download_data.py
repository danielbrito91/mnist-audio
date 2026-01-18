import subprocess
from pathlib import Path

from dotenv import load_dotenv

_ = load_dotenv()


MNIST_KAGGLE = 'sripaadsrinivasan/audio-mnist'
if __name__ == '__main__':
    if not Path('data/raw').exists():
        Path('data/raw').mkdir(parents=True, exist_ok=True)

    # Download data from Kaggle via CLI
    subprocess.run([
        'kaggle',
        'datasets',
        'download',
        '-d',
        MNIST_KAGGLE,
        '-p',
        'data/raw',
    ])
    print('Data downloaded successfully')
    print(f'Data saved to {Path("data/raw")}')

    # Unzip data
    subprocess.run([
        'unzip',
        'data/raw/audio-mnist.zip',
        '-d',
        'data/raw',
    ])
    print('Data unzipped successfully')
    print(f'Data saved to {Path("data/raw")}')

    # Remove zip file
    subprocess.run([
        'rm',
        'data/raw/audio-mnist.zip',
    ])
    print('Zip file removed successfully')
    print(f'Zip file removed from {Path("data/raw")}')
