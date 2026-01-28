# GAN Configuration Variables Guide

A comprehensive guide to understanding and configuring your GAN training parameters.

---

## Table of Contents
1. [IMAGE_FOLDER](#1-image_folder)
2. [IMG_SIZE](#2-img_size)
3. [IMG_CHANNELS](#3-img_channels)
4. [EPOCHS](#4-epochs)
5. [BATCH_SIZE](#5-batch_size)
6. [Practical Examples](#practical-example-scenarios)
7. [Quick Recommendations](#quick-recommendations)

---

## 1. IMAGE_FOLDER

```python
IMAGE_FOLDER = "./training_images"
```

### What it does
Points to the folder containing your training images.

### How it affects results
- The GAN learns to generate images **similar to what's in this folder**
- If you put cat photos → it generates cats
- If you put faces → it generates faces
- If you put landscapes → it generates landscapes

### Examples
```python
IMAGE_FOLDER = "./cat_images"      # Trains on cats
IMAGE_FOLDER = "./my_artwork"      # Trains on your art style
IMAGE_FOLDER = "./product_photos"  # Trains on products
```

### Important Notes
- All images in the folder will be used for training
- Supported formats: JPG, PNG, JPEG, BMP, GIF
- Mixed content leads to poor results

---

## 2. IMG_SIZE

```python
IMG_SIZE = 64  # Change to 128, 256, etc.
```

### What it does
Resizes ALL your images to this dimension (creates square images: 64x64, 128x128, etc.)

### Impact Comparison

| Size | Quality | Speed | Memory | Best For |
|------|---------|-------|---------|----------|
| 64x64 | Lower detail | Very Fast ⚡ | Low 💾 | Quick experiments, simple images |
| 128x128 | Good detail | Medium ⚡⚡ | Medium 💾💾 | Balanced quality/speed |
| 256x256 | High detail | Slow ⚡⚡⚡ | High 💾💾💾 | Professional results, complex images |
| 512x512 | Very high detail | Very Slow ⚡⚡⚡⚡ | Very High 💾💾💾💾 | Best quality (needs powerful GPU) |

### Real-World Impact
- **64px**: Good for icons, simple objects, testing
- **128px**: Good for faces, small objects with detail
- **256px**: Good for detailed portraits, complex scenes
- **512px+**: Professional photography quality

### ⚠️ Warning
Larger sizes need more GPU memory and training time!

---

## 3. IMG_CHANNELS

```python
IMG_CHANNELS = 3  # 3 for RGB, 1 for grayscale
```

### What it does
Defines the color mode of images.

### Options

#### RGB (Color) - `IMG_CHANNELS = 3`
- Uses Red, Green, Blue channels
- Generates color images
- Learns color patterns and relationships
- Requires more computational power

#### Grayscale (Black & White) - `IMG_CHANNELS = 1`
- Single intensity channel
- Generates black and white images
- Trains faster (less data to process)
- Lower memory requirements

### When to Use What

**Use `IMG_CHANNELS = 3` for:**
- Color photographs
- Artwork and paintings
- Product images
- Any colored content

**Use `IMG_CHANNELS = 1` for:**
- Black and white photos
- Sketches and drawings
- X-rays or medical images
- Historical photographs
- Faster training experiments

---

## 4. EPOCHS

```python
EPOCHS = 100  # Training duration
```

### What it does
Number of times the model sees your entire dataset during training.

### Quality vs Epochs

| Epochs | Result Quality | Description |
|--------|---------------|-------------|
| 10-30 | Poor ⭐ | Blurry, incomplete learning |
| 50-100 | Decent ⭐⭐⭐ | Recognizable shapes, basic features |
| 100-200 | Good ⭐⭐⭐⭐ | Clear images, realistic details |
| 200-500 | Excellent ⭐⭐⭐⭐⭐ | Best quality, fine details |
| 500+ | Risk of Overfitting ⚠️ | May memorize training images |

### Training Time Estimates

**Small dataset (100 images):**
- 1 epoch ≈ 5-10 seconds
- 100 epochs ≈ 8-15 minutes

**Medium dataset (1,000 images):**
- 1 epoch ≈ 30-60 seconds
- 100 epochs ≈ 50-100 minutes

**Large dataset (10,000 images):**
- 1 epoch ≈ 2-5 minutes
- 100 epochs ≈ 3-8 hours

### Finding the Right Number
- **Too few epochs** → Generator produces noise/blur
- **Right amount** → Generator produces realistic images
- **Too many epochs** → May start copying training images exactly (overfitting)

### 💡 Tip
Monitor the generated images every 10-20 epochs. Stop when quality plateaus or starts degrading.

---

## 5. BATCH_SIZE

```python
BATCH_SIZE = 32  # Adjust based on GPU memory
```

### What it does
Number of images processed together in one training step.

### Impact Comparison

| Batch Size | Training Speed | Memory Use | Quality | Best For |
|------------|---------------|------------|---------|----------|
| 8 | Slower 🐌 | Low 💾 | More stable | Large images (256px+), weak GPU |
| 16 | Medium 🐇 | Medium 💾💾 | Balanced | Most cases |
| 32 | Faster 🚀 | High 💾💾💾 | Good | Small images (64px), strong GPU |
| 64+ | Fastest ⚡ | Very High 💾💾💾💾 | Best | Powerful GPU, small images |

### Trade-offs

**Larger Batch Size:**
- ✅ Faster training
- ✅ More stable gradient updates
- ❌ Requires more GPU memory
- ❌ May need powerful hardware

**Smaller Batch Size:**
- ✅ Works on weaker hardware
- ✅ Lower memory requirements
- ❌ Slower training
- ❌ Less stable training (more noisy)

### GPU Memory Guide

```
Image Size      Recommended Batch Size    GPU Memory Needed
─────────────────────────────────────────────────────────────
64x64           32-64                     ~4GB
128x128         16-32                     ~6GB
256x256         4-16                      ~8GB+
512x512         2-8                       ~12GB+
```

### ⚠️ Common Error
If you see **"CUDA out of memory"** or **"Out of Memory"** error:
1. Reduce `BATCH_SIZE` (try halving it)
2. Or reduce `IMG_SIZE`
3. Or close other applications using GPU

---

## How Your Images Affect Training

### Number of Images

| Image Count | Quality | Description |
|-------------|---------|-------------|
| 10-50 | Poor | Will overfit (copy images exactly) |
| 100-500 | Decent | Basic learning, limited variety |
| 1,000-5,000 | Good | Solid results, good variety |
| 10,000+ | Excellent | Professional quality, great diversity |

### Image Quality Requirements

**✅ Good Training Data:**
- Consistent style (all photos OR all drawings)
- Similar subjects (all faces, all cars, all landscapes)
- Good image quality (not blurry or corrupted)
- Proper lighting and framing

**❌ Poor Training Data:**
- Mixed styles (photos + cartoons + sketches)
- Random unrelated subjects (faces + cars + trees)
- Low quality or corrupted images
- Inconsistent aspect ratios

---

## Practical Example Scenarios

### Scenario 1: Quick Test with Cat Photos

```python
IMAGE_FOLDER = "./my_cats"
IMG_SIZE = 64          # Fast training
IMG_CHANNELS = 3       # Color cats
EPOCHS = 50            # Quick results
BATCH_SIZE = 32        # Fast processing
```

**Expected Results:**
- ⏱️ Training time: ~10-15 minutes
- 🎨 Output: Blurry but recognizable cat-like images
- 💾 GPU memory: ~4GB
- 👍 Good for: Quick experimentation

---

### Scenario 2: High-Quality Face Generation

```python
IMAGE_FOLDER = "./face_photos"
IMG_SIZE = 128         # Good detail for faces
IMG_CHANNELS = 3       # Color faces
EPOCHS = 200           # Better quality
BATCH_SIZE = 16        # Balanced
```

**Expected Results:**
- ⏱️ Training time: ~1-2 hours
- 🎨 Output: Clear, detailed realistic faces
- 💾 GPU memory: ~6GB
- 👍 Good for: Face generation projects

---

### Scenario 3: Artistic Sketches (Black & White)

```python
IMAGE_FOLDER = "./sketches"
IMG_SIZE = 256         # High detail
IMG_CHANNELS = 1       # Grayscale only
EPOCHS = 150           # Good quality
BATCH_SIZE = 8         # Lower for big images
```

**Expected Results:**
- ⏱️ Training time: ~2-3 hours
- 🎨 Output: Detailed sketch-style images
- 💾 GPU memory: ~8GB
- 👍 Good for: Artistic/sketch generation

---

### Scenario 4: Professional Landscape Photos

```python
IMAGE_FOLDER = "./landscapes"
IMG_SIZE = 256         # High detail needed
IMG_CHANNELS = 3       # Full color
EPOCHS = 300           # Best quality
BATCH_SIZE = 8         # Manageable memory
```

**Expected Results:**
- ⏱️ Training time: ~4-6 hours
- 🎨 Output: High-quality landscape images
- 💾 GPU memory: ~8-10GB
- 👍 Good for: Professional work

---

## Quick Recommendations

### 🔰 For Beginners (Your First Try)

```python
IMAGE_FOLDER = "./training_images"
IMG_SIZE = 64
IMG_CHANNELS = 3
EPOCHS = 50
BATCH_SIZE = 32
```

**Why:** Fast training, quick results, low memory usage

---

### 🎯 For Good Quality Results

```python
IMAGE_FOLDER = "./training_images"
IMG_SIZE = 128
IMG_CHANNELS = 3
EPOCHS = 150
BATCH_SIZE = 16
```

**Why:** Balanced quality/speed, works on most GPUs

---

### 🏆 For Best Quality (Requires Good GPU)

```python
IMAGE_FOLDER = "./training_images"
IMG_SIZE = 256
IMG_CHANNELS = 3
EPOCHS = 200
BATCH_SIZE = 8
```

**Why:** Professional results, high detail

---

## Progressive Training Strategy

Start small and scale up gradually:

### Step 1: Validation Run
```python
IMG_SIZE = 64
EPOCHS = 20
BATCH_SIZE = 32
```
**Goal:** Verify your images load correctly and training works

### Step 2: Quality Check
```python
IMG_SIZE = 128
EPOCHS = 100
BATCH_SIZE = 16
```
**Goal:** See if results are promising

### Step 3: Final Production
```python
IMG_SIZE = 256
EPOCHS = 200
BATCH_SIZE = 8
```
**Goal:** Generate best quality outputs

---

## Troubleshooting Guide

### Problem: Training is too slow
**Solutions:**
- ✅ Reduce `IMG_SIZE`
- ✅ Increase `BATCH_SIZE` (if you have GPU memory)
- ✅ Reduce `EPOCHS`

### Problem: Out of memory error
**Solutions:**
- ✅ Reduce `BATCH_SIZE` (try 16 → 8 → 4)
- ✅ Reduce `IMG_SIZE`
- ✅ Use `IMG_CHANNELS = 1` instead of 3

### Problem: Generated images are blurry
**Solutions:**
- ✅ Increase `EPOCHS` (try 100 → 200)
- ✅ Check if you have enough training images (need 500+)
- ✅ Ensure training images are high quality

### Problem: Generated images look exactly like training images
**Solutions:**
- ✅ Reduce `EPOCHS` (you're overfitting)
- ✅ Add more diverse training images
- ✅ Reduce training time

### Problem: Results are noisy/random
**Solutions:**
- ✅ Increase `EPOCHS`
- ✅ Check training data consistency
- ✅ Ensure enough training images (100+ minimum)

---

## Hardware Requirements Guide

### Minimum Requirements
- **CPU:** Multi-core processor (4+ cores)
- **RAM:** 8GB+
- **GPU:** 4GB VRAM (NVIDIA recommended)
- **Storage:** 10GB+ free space

### Recommended for Good Results
- **CPU:** Modern multi-core (8+ cores)
- **RAM:** 16GB+
- **GPU:** 8GB VRAM (NVIDIA RTX series)
- **Storage:** 50GB+ SSD

### Professional Setup
- **CPU:** High-end multi-core (16+ cores)
- **RAM:** 32GB+
- **GPU:** 12GB+ VRAM (NVIDIA RTX 3080/4080 or better)
- **Storage:** 100GB+ NVMe SSD

---

## Final Tips

1. **Start small:** Always begin with small `IMG_SIZE` and `EPOCHS` to test
2. **Monitor progress:** Check generated samples every 10-20 epochs
3. **Save checkpoints:** The code saves models automatically
4. **Be patient:** Quality results take time (hours, not minutes)
5. **Experiment:** Try different combinations to find what works best
6. **Quality over quantity:** 500 good images > 5000 random images

---

## Additional Resources

- For more advanced configurations, modify the code directly
- Monitor GPU usage with: `nvidia-smi` (Linux/Windows) or Activity Monitor (Mac)
- Consider using cloud GPUs (Google Colab, AWS) for large-scale training

---

**Last Updated:** January 2026  
**Version:** 1.0