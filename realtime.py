import cv2
import torch
import numpy as np
from torchvision import transforms, models
from PIL import Image

# --------------------------
# 1️⃣ Load model
# --------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"✅ Using device: {device}")

model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(model.last_channel, 2)

# Load your trained model weights
model.load_state_dict(torch.load('models/mask_detector.pth', map_location=device))
model.to(device)
model.eval()

# --------------------------
# 2️⃣ Define transform
# --------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    #transforms.Normalize([0.485, 0.456, 0.406],
    #                     [0.229, 0.224, 0.225])  # standard ImageNet normalization
])

# --------------------------
# 3️⃣ Labels
# --------------------------
classes = ['with_mask', 'without_mask']

# --------------------------
# 4️⃣ Initialize webcam
# --------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam")
    exit()

print("🎥 Press 'q' to quit")

# --------------------------
# 5️⃣ Real-time detection loop
# --------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB (OpenCV uses BGR)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image for transforms
    pil_img = Image.fromarray(rgb)

    # Apply transforms and send to device
    tensor = transform(pil_img).unsqueeze(0).to(device)

    # Model prediction
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
        label = classes[pred.item()]
        confidence = conf.item() * 100

    # Choose bounding box color
    color = (0, 255, 0) if label == 'with_mask' else (0, 0, 255)

    # Display label and confidence
    text = f"{label}: {confidence:.1f}%"
    cv2.putText(frame, text, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Show frame
    cv2.imshow("Mask Detection (PyTorch)", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Exited cleanly.")
