# Deep Learning Models for CIFAR-100 Classification and Generation

This project involved the design, training, and evaluation of two deep learning models using the **CIFAR-100** dataset: a **discriminative model** for classification and a **generative model** for synthesizing images. The key challenges were to meet strict limitations on the number of parameters and optimization steps while achieving high performance in both tasks.

## Project Overview

1. **Deep Discriminative Model**:
   - The goal was to design a classifier that predicts the class of CIFAR-100 images while adhering to a strict constraint of fewer than 100,000 parameters.
   - I trained the model for no more than 10,000 optimization steps to ensure optimal efficiency.

2. **Deep Generative Model**:
   - The generative model aimed to synthesize realistic images from the CIFAR-100 dataset with a parameter limit of fewer than 1,000,000 parameters.
   - The model was trained for no more than 50,000 optimization steps, and the output was evaluated based on realism, diversity, and uniqueness.

## Key Skills and Techniques Learned

### 1. Designing a Deep Neural Network Architecture
   - **Architecture Design**: I designed custom deep neural network architectures for both the discriminative and generative models, ensuring they adhered to the parameter constraints.
   - **Layer Selection**: For the discriminative model, I explored various combinations of convolutional layers, pooling layers, and fully connected layers. For the generative model, I implemented architectures such as **autoencoders** or **variational autoencoders (VAEs)**, ensuring that the networks could synthesize realistic images without exceeding the parameter limit.

### 2. Model Training and Optimization
   - **Training for Classification**: I implemented a **classification pipeline** using **PyTorch** that trained the discriminative model on CIFAR-100. I carefully managed the number of optimization steps (≤10,000) to avoid penalties, and I tracked the training and testing accuracy at each epoch.
   - **Loss Functions and Optimizers**: For the classification task, I used **cross-entropy loss** with **Stochastic Gradient Descent (SGD)**, while for the generative task, I used a **loss function** appropriate for the generative model (such as **binary cross-entropy** or **mean squared error**), paired with optimizers like **Adam**.
   - **Training Efficiency**: I focused on optimizing both the **training speed** and **memory usage** to stay within the constraints on parameter count and optimization steps.

### 3. Model Evaluation and Results Reporting
   - **Accuracy Tracking**: I implemented code to track the training and testing accuracy over the course of training, and I used this data to produce plots that demonstrate how the model performed on the CIFAR-100 test set.
   - **Loss Reporting**: After training the discriminative model, I reported the final **training loss**, **training accuracy**, and **test accuracy**, including both **means and standard deviations**.
   - **FID Scores**: For the generative model, I calculated the **Fréchet Inception Distance (FID)** between 10,000 generated images and 10,000 test images, providing a quantitative measure of the quality of the generated samples.
   - **Visual Evaluation**: I generated visual results such as:
     - A batch of **64 unique generated images**.
     - **Interpolations** between 8 pairs of generated images.
     - **Sample comparison** with real images from the CIFAR-100 test set.

### 4. Computational Efficiency and Resource Management
   - **Memory and Parameter Management**: In both models, I ensured that the architectures were lightweight by limiting the number of parameters (for the discriminative model: ≤100,000 parameters; for the generative model: ≤1,000,000 parameters).
   - **Optimization Step Constraints**: I carefully monitored the optimization steps to stay within the limits (≤10,000 steps for the discriminative model and ≤50,000 steps for the generative model).
   - **Efficient Training**: To optimize performance without exceeding the parameter limits, I experimented with various architectures and training configurations, balancing model complexity with computational resources.

### 5. Deep Generative Models and Sampling
   - **Generative Model Training**: I trained a **deep generative model** (such as a **variational autoencoder** or **GAN**) to synthesize realistic CIFAR-100 images from noise vectors. This involved sampling from a prior distribution and generating novel images that resembled the original CIFAR-100 dataset.
   - **Latent Space Exploration**: I utilized the latent space of the generative model to generate diverse samples and interpolate between images to observe the model's ability to generate continuous variations.
   - **GAN (if applicable)**: If a **Generative Adversarial Network (GAN)** was used, I implemented both the **generator** and **discriminator** networks and ensured that their combined parameters stayed within the model size constraints.

### 6. Data Handling and Preprocessing
   - **CIFAR-100 Dataset**: I utilized the CIFAR-100 dataset, which consists of 100 classes, each with 600 images. I performed necessary **preprocessing steps**, such as **normalization**, **augmentation**, and **shuffling**, to ensure the models trained effectively.
   - **Class Label Utilization**: For the generative model, I used the CIFAR-100 **class labels** to condition the sampling process, allowing the model to generate diverse images while still maintaining structure in the generated samples.

## Technical Deliverables

1. **Discriminative Model**:
   - A classifier trained on CIFAR-100 with fewer than 100,000 parameters and less than 10,000 optimization steps.
   - Plots of training and test accuracy over time.
   - Final training loss, training accuracy, and test accuracy (with means and standard deviations).

2. **Generative Model**:
   - A model capable of generating diverse and realistic images with fewer than 1,000,000 parameters and less than 50,000 optimization steps.
   - A batch of 64 non-cherry-picked generated images.
   - Interpolations between 8 pairs of generated images.
   - FID scores comparing the generated images to the CIFAR-100 test dataset.

## Summary

This project provided hands-on experience in designing and training deep learning models for both classification and generative tasks on the CIFAR-100 dataset. Key skills acquired include:
- **Model Architecture Design**: Creating deep neural networks that meet strict parameter and optimization constraints.
- **Training and Optimization**: Managing training processes to optimize performance while adhering to computational limits.
- **Generative Modeling**: Implementing and training generative models like VAEs and GANs for image synthesis.
- **Model Evaluation**: Assessing model performance with various metrics, including accuracy, FID scores, and visual samples.

These skills are highly transferable to roles involving deep learning, computer vision, and generative modeling, and demonstrate the ability to balance model performance with computational efficiency.
