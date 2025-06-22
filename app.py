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


@app.route('/api/generate', methods=['POST'])
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


@app.route('/api/generate_batch', methods=['POST'])
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


@app.route('/api/health')
def health():
    return jsonify({"status": "healthy"})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5100))
    app.run(host='0.0.0.0', port=port, debug=False)
