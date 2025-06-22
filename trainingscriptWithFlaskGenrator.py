# ========================================
# PART 1: COLAB TRAINING SCRIPT
# ========================================

# Run this in Google Colab for training
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from google.colab import files
import zipfile

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Optimized hyperparameters for T4 GPU
batch_size = 128  # Increased for better T4 utilization
learning_rate = 0.0002
beta1 = 0.5
num_epochs = 25  # Reduced slightly for faster training
noise_dim = 100
num_classes = 10
image_size = 28


# Generator Network (Optimized)
class Generator(nn.Module):
    def __init__(self, noise_dim, num_classes, img_size):
        super(Generator, self).__init__()
        self.img_size = img_size

        # Label embedding
        self.label_emb = nn.Embedding(num_classes, 50)

        # Main network
        self.model = nn.Sequential(
            nn.Linear(noise_dim + 50, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),

            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),

            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, img_size * img_size),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_emb = self.label_emb(labels)
        gen_input = torch.cat([noise, label_emb], dim=1)
        img = self.model(gen_input)
        return img.view(img.size(0), 1, self.img_size, self.img_size)


# Discriminator Network (Optimized)
class Discriminator(nn.Module):
    def __init__(self, num_classes, img_size):
        super(Discriminator, self).__init__()

        # Label embedding
        self.label_emb = nn.Embedding(num_classes, 50)

        self.model = nn.Sequential(
            nn.Linear(img_size * img_size + 50, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        img_flat = img.view(img.size(0), -1)
        label_emb = self.label_emb(labels)
        disc_input = torch.cat([img_flat, label_emb], dim=1)
        return self.model(disc_input)


# Data loading (optimized for Colab)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=2, pin_memory=True  # Optimized for GPU
)

# Initialize networks
G = Generator(noise_dim, num_classes, image_size).to(device)
D = Discriminator(num_classes, image_size).to(device)


# Initialize weights
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0.0, 0.02)


G.apply(weights_init)
D.apply(weights_init)

# Loss and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=learning_rate, betas=(beta1, 0.999))


# Training with progress tracking
def train_cgan():
    G.train()
    D.train()

    g_losses, d_losses = [], []

    for epoch in range(num_epochs):
        epoch_g_loss = epoch_d_loss = 0

        for i, (real_imgs, labels) in enumerate(train_loader):
            batch_size_curr = real_imgs.size(0)
            real_imgs, labels = real_imgs.to(device), labels.to(device)

            # Labels for loss
            real_labels = torch.ones(batch_size_curr, 1, device=device)
            fake_labels = torch.zeros(batch_size_curr, 1, device=device)

            # Train Discriminator
            optimizer_D.zero_grad()

            # Real images
            d_real = D(real_imgs, labels)
            d_real_loss = criterion(d_real, real_labels)

            # Fake images
            noise = torch.randn(batch_size_curr, noise_dim, device=device)
            fake_labels_gen = torch.randint(0, num_classes, (batch_size_curr,), device=device)
            fake_imgs = G(noise, fake_labels_gen)
            d_fake = D(fake_imgs.detach(), fake_labels_gen)
            d_fake_loss = criterion(d_fake, fake_labels)

            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            d_fake = D(fake_imgs, fake_labels_gen)
            g_loss = criterion(d_fake, real_labels)
            g_loss.backward()
            optimizer_G.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()

            if i % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}] Step [{i}/{len(train_loader)}] '
                      f'D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}')

        g_losses.append(epoch_g_loss / len(train_loader))
        d_losses.append(epoch_d_loss / len(train_loader))

        # Show sample every 5 epochs
        if (epoch + 1) % 5 == 0:
            show_samples(epoch + 1)

    return g_losses, d_losses


def show_samples(epoch):
    G.eval()
    with torch.no_grad():
        sample_labels = torch.arange(0, 10, device=device)
        sample_noise = torch.randn(10, noise_dim, device=device)
        fake_imgs = G(sample_noise, sample_labels)
        fake_imgs = (fake_imgs + 1) / 2

        plt.figure(figsize=(15, 2))
        for i in range(10):
            plt.subplot(1, 10, i + 1)
            plt.imshow(fake_imgs[i][0].cpu(), cmap='gray')
            plt.title(f'{i}')
            plt.axis('off')
        plt.suptitle(f'Epoch {epoch}')
        plt.show()
    G.train()


# Main training execution
print("Starting training...")
g_losses, d_losses = train_cgan()

# Final samples
show_samples("Final")

# Save model for deployment
os.makedirs('model_files', exist_ok=True)

# Save generator only (smaller file for deployment)
torch.save({
    'model_state_dict': G.state_dict(),
    'noise_dim': noise_dim,
    'num_classes': num_classes,
    'image_size': image_size
}, 'model_files/cgan_generator.pth')

# Save model architecture as well
torch.save(G, 'model_files/cgan_generator_full.pth')

print("Training completed! Models saved.")


# Create deployment package
def create_deployment_package():
    """Create a zip file with all necessary files for deployment"""

    # Create Flask app file
    flask_app_code = '''
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import base64
from PIL import Image
import os

app = Flask(__name__)
CORS(app)

# Generator class (same as training)
class Generator(nn.Module):
    def __init__(self, noise_dim, num_classes, img_size):
        super(Generator, self).__init__()
        self.img_size = img_size

        self.label_emb = nn.Embedding(num_classes, 50)

        self.model = nn.Sequential(
            nn.Linear(noise_dim + 50, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),

            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),

            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, img_size * img_size),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_emb = self.label_emb(labels)
        gen_input = torch.cat([noise, label_emb], dim=1)
        img = self.model(gen_input)
        return img.view(img.size(0), 1, self.img_size, self.img_size)

# Load model
device = torch.device('cpu')  # Use CPU for deployment
checkpoint = torch.load('cgan_generator.pth', map_location=device)

generator = Generator(
    checkpoint['noise_dim'], 
    checkpoint['num_classes'], 
    checkpoint['image_size']
)
generator.load_state_dict(checkpoint['model_state_dict'])
generator.eval()

@app.route('/')
def home():
    return jsonify({
        "message": "MNIST cGAN API is running!",
        "endpoints": {
            "/generate": "POST - Generate digit image",
            "/generate_batch": "POST - Generate multiple digits"
        }
    })

@app.route('/generate', methods=['POST'])
def generate_digit():
    try:
        data = request.get_json()
        digit = int(data.get('digit', 0))

        if digit < 0 or digit > 9:
            return jsonify({"error": "Digit must be between 0 and 9"}), 400

        # Generate image
        with torch.no_grad():
            noise = torch.randn(1, checkpoint['noise_dim'])
            label = torch.tensor([digit])
            fake_img = generator(noise, label)

            # Convert to PIL Image
            img_array = fake_img[0][0].numpy()
            img_array = (img_array + 1) / 2  # Normalize to [0, 1]
            img_array = (img_array * 255).astype(np.uint8)

            # Convert to base64
            img = Image.fromarray(img_array, 'L')
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_str = base64.b64encode(img_buffer.getvalue()).decode()

        return jsonify({
            "digit": digit,
            "image": f"data:image/png;base64,{img_str}",
            "success": True
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate_batch', methods=['POST'])
def generate_batch():
    try:
        data = request.get_json()
        digits = data.get('digits', list(range(10)))

        if not isinstance(digits, list) or len(digits) > 20:
            return jsonify({"error": "Digits should be a list with max 20 items"}), 400

        results = []

        with torch.no_grad():
            for digit in digits:
                if 0 <= digit <= 9:
                    noise = torch.randn(1, checkpoint['noise_dim'])
                    label = torch.tensor([digit])
                    fake_img = generator(noise, label)

                    img_array = fake_img[0][0].numpy()
                    img_array = (img_array + 1) / 2
                    img_array = (img_array * 255).astype(np.uint8)

                    img = Image.fromarray(img_array, 'L')
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format='PNG')
                    img_str = base64.b64encode(img_buffer.getvalue()).decode()

                    results.append({
                        "digit": digit,
                        "image": f"data:image/png;base64,{img_str}"
                    })

        return jsonify({
            "results": results,
            "count": len(results),
            "success": True
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
'''

    # Create requirements.txt
    requirements = '''torch==2.0.1
torchvision==0.15.2
flask==2.3.3
flask-cors==4.0.0
Pillow==10.0.0
numpy==1.24.3
gunicorn==21.2.0'''

    # Create Procfile for Render
    procfile = 'web: gunicorn app:app'

    # Create render.yaml
    render_yaml = '''services:
  - type: web
    name: mnist-cgan-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app
    envVars:
      - key: PYTHON_VERSION
        value: 3.9.16'''

    # Write files
    with open('model_files/app.py', 'w') as f:
        f.write(flask_app_code)

    with open('model_files/requirements.txt', 'w') as f:
        f.write(requirements)

    with open('model_files/Procfile', 'w') as f:
        f.write(procfile)

    with open('model_files/render.yaml', 'w') as f:
        f.write(render_yaml)

    # Create README for deployment
    readme = '''# MNIST cGAN Flask API

## Deployment on Render

1. Upload all files to a GitHub repository
2. Connect your GitHub repo to Render
3. Deploy as a Web Service

## API Endpoints

### Generate Single Digit
POST /generate
```json
{
    "digit": 5
}
```

### Generate Multiple Digits  
POST /generate_batch
```json
{
    "digits": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}
```

## Local Testing
```bash
pip install -r requirements.txt
python app.py
```

The API will be available at http://localhost:5000
'''

    with open('model_files/README.md', 'w') as f:
        f.write(readme)

    # Create zip file
    with zipfile.ZipFile('deployment_package.zip', 'w') as zipf:
        for root, dirs, files in os.walk('model_files'):
            for file in files:
                zipf.write(os.path.join(root, file), file)

    print("Deployment package created: deployment_package.zip")


# Create the deployment package
create_deployment_package()

# Download files
print("\\nDownloading files...")
files.download('deployment_package.zip')
files.download('model_files/cgan_generator.pth')

print("\\n=== DEPLOYMENT INSTRUCTIONS ===")
print("1. Extract deployment_package.zip")
print("2. Upload all files to a GitHub repository")
print("3. Connect GitHub repo to Render")
print("4. Deploy as Web Service")
print("5. Your API will be live!")

# ========================================
# PART 2: EXAMPLE API USAGE
# ========================================

# Use this code to test your deployed API
test_api_code = '''
import requests
import json
from PIL import Image
import base64
import io

# Replace with your deployed API URL
API_URL = "https://your-app-name.onrender.com"

def test_single_generation():
    """Test generating a single digit"""
    url = f"{API_URL}/generate"
    data = {"digit": 7}

    response = requests.post(url, json=data)

    if response.status_code == 200:
        result = response.json()
        print(f"Generated digit: {result['digit']}")

        # Decode and save image
        img_data = result['image'].split(',')[1]  # Remove data:image/png;base64,
        img_bytes = base64.b64decode(img_data)
        img = Image.open(io.BytesIO(img_bytes))
        img.save(f"generated_digit_{result['digit']}.png")
        print("Image saved!")
    else:
        print(f"Error: {response.text}")

def test_batch_generation():
    """Test generating multiple digits"""
    url = f"{API_URL}/generate_batch"
    data = {"digits": [0, 1, 2, 3, 4]}

    response = requests.post(url, json=data)

    if response.status_code == 200:
        result = response.json()
        print(f"Generated {result['count']} images")

        for i, item in enumerate(result['results']):
            img_data = item['image'].split(',')[1]
            img_bytes = base64.b64decode(img_data)
            img = Image.open(io.BytesIO(img_bytes))
            img.save(f"batch_digit_{item['digit']}.png")
        print("All images saved!")
    else:
        print(f"Error: {response.text}")

# Run tests
if __name__ == "__main__":
    test_single_generation()
    test_batch_generation()
'''

print("\\n=== API TESTING CODE ===")
print("Use this code to test your deployed API:")
print(test_api_code)