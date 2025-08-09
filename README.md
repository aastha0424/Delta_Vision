# Change Detection using ChangeStar (ResNet-18)

This project demonstrates how to perform **change detection** between two images taken at different times using the **ChangeStar** deep learning architecture, implemented in PyTorch.

The notebook (`change_detection.ipynb`) walks through:
- Loading a ChangeStar model (1x96) based on ResNet-18.
- Preparing bi-temporal image data (T1 and T2 images).
- Running inference to detect changes.
- Visualizing change maps.

---

## 🔧 Requirements

Install the required Python packages:

```bash
pip install torch torchvision matplotlib pillow numpy
```

---

## 🚀 Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Create `requirements.txt` with the dependencies listed above)*

3. **Run the notebook:**
   ```bash
   jupyter notebook change_detection.ipynb
   ```

4. **Steps inside the notebook:**
   - Load or simulate two input images (T1 and T2).
   - Concatenate them for model input.
   - Load the ChangeStar model.
   - Perform inference.
   - Visualize the predicted change map.

---

## 📸 Example Workflow

```python
import torch

# Simulated images
t1_image = torch.rand(1, 3, 512, 512)
t2_image = torch.rand(1, 3, 512, 512)

# Concatenate along the channel dimension
bi_images = torch.cat([t1_image, t2_image], dim=1)

# Pass through model (example)
output = model(bi_images)
```

---

## 📚 References

- [ChangeStar: A Universal Architecture for Change Detection](https://arxiv.org/abs/2103.00284)
- [PyTorch Documentation](https://pytorch.org/)

---

## 📝 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
