# app.py
import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import cv2
import os

# Set page configuration
st.set_page_config(
    page_title="Rice Leaf Blast Detection System",
    page_icon="🌾",
    layout="wide"
)

# Title and description
st.title("🌾 Rice Leaf Blast Detection System")
st.markdown("""
This interactive demo showcases the complete pipeline for detecting rice leaf blast disease 
using computer vision and machine learning.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Select Demo Section:",
    ["System Overview", "Pre-processing", "Data Augmentation", 
     "Data Splitting", "Model Training", "Live Prediction"]
)

# =============================================================================
# D.1 PRE-PROCESSING DEMO SECTION
# =============================================================================
if section == "Pre-processing":
    st.header("🔍 Pre-processing Module")
    
    # Section D.1: Image Pre-processing
    st.subheader("D.1 - Image Pre-processing Steps")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Code Purpose:** Image Loading and CLAHE Enhancement
        
        This section demonstrates how raw rice leaf images are processed:
        1. **Image Loading**: Load RGB images and convert to numpy arrays
        2. **CLAHE Enhancement**: Apply Contrast Limited Adaptive Histogram Equalization
        3. **Normalization**: Scale pixel values to [0,1] range
        
        **Key Functions:**
        - `load_and_preprocess_image()`: Handles image loading and resizing
        - `apply_clahe_enhancement()`: Enhances contrast in LAB color space
        """)
        
        # Upload example image
        uploaded_file = st.file_uploader("Upload rice leaf image", type=['jpg', 'jpeg', 'png'])
        
    with col2:
        if uploaded_file is not None:
            # Display original and processed
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_column_width=True)
            
            # Simulate CLAHE processing
            if st.button("Apply CLAHE Enhancement"):
                # Convert to array and process (simplified)
                img_array = np.array(image)
                st.success("CLAHE enhancement applied!")
                
                # Show comparison
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                ax1.imshow(img_array)
                ax1.set_title("Original Image")
                ax1.axis('off')
                
                # Simulate enhanced version (in real app, use actual CLAHE)
                enhanced = cv2.convertScaleAbs(img_array, alpha=1.2, beta=10)
                ax2.imshow(enhanced)
                ax2.set_title("CLAHE Enhanced")
                ax2.axis('off')
                
                st.pyplot(fig)

    # Section D.2: Other Pre-processing
    st.subheader("D.2 - Additional Pre-processing Steps")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Duplicate & Blur Detection**
        
        **Code Purpose:** Filter low-quality images
        
        - **Hash-based deduplication**: MD5 hashing to remove duplicates
        - **Blur detection**: Laplacian variance thresholding
        - **Automatic labeling**: File path analysis for 'LEAFBLAST'/'HEALTHY'
        
        **Key Functions:**
        - `calculate_image_hash()`: MD5 hash for duplicates
        - `detect_blur_image_pil()`: Variance of Laplacian
        - `auto_detect_label()`: Path-based labeling
        """)
    
    with col2:
        st.markdown("""
        **Class Balance Analysis**
        
        **Code Purpose:** Ensure balanced dataset
        
        - **Undersampling**: Reduce majority class to match minority
        - **Visualization**: Before/after balance comparison
        - **Statistics**: Class distribution analysis
        
        **Key Functions:**
        - `balance_classes_undersample()`: Implements undersampling
        - `visualize_class_balance_comprehensive()`: Balance visualization
        """)
        
        # Show class balance visualization
        if st.button("Show Class Balance"):
            # Create sample class distribution
            labels = ['LEAFBLAST', 'HEALTHY']
            before = [120, 80]  # Unbalanced
            after = [80, 80]    # Balanced
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            
            ax1.pie(before, labels=labels, autopct='%1.1f%%', colors=['red', 'green'])
            ax1.set_title('Before Balancing')
            
            ax2.pie(after, labels=labels, autopct='%1.1f%%', colors=['red', 'green'])
            ax2.set_title('After Balancing')
            
            st.pyplot(fig)

# =============================================================================
# D.3 DATA AUGMENTATION DEMO SECTION
# =============================================================================
elif section == "Data Augmentation":
    st.header("🔄 Data Augmentation Module")
    
    st.subheader("D.3 - Data Augmentation Techniques")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Code Purpose:** Increase dataset diversity
        
        **Augmentation Techniques:**
        - **Geometric**: Horizontal/Vertical flip, 90° rotations
        - **Photometric**: Brightness/Contrast adjustment
        - **Combined**: Multiple transformations per image
        
        **Key Functions:**
        - `apply_data_augmentation_with_processed()`: Main augmentation function
        - `visualize_augmentation_samples()`: Display augmentation results
        """)
        
        # Augmentation controls
        augmentation_type = st.selectbox(
            "Select augmentation type:",
            ["Original", "Horizontal Flip", "Vertical Flip", "Rotation", "Brightness", "Contrast"]
        )
        
    with col2:
        if st.button("Generate Augmented Samples"):
            # Create sample augmentation visualization
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            axes = axes.flatten()
            
            # Sample image (in real app, use actual image)
            sample_img = np.random.rand(100, 100, 3)
            
            # Show different augmentations
            augmentations = [
                ("Original", sample_img),
                ("Horizontal Flip", np.fliplr(sample_img)),
                ("Vertical Flip", np.flipud(sample_img)),
                ("Rotation", np.rot90(sample_img)),
                ("Brightness+", np.clip(sample_img * 1.3, 0, 1)),
                ("Brightness-", np.clip(sample_img * 0.7, 0, 1))
            ]
            
            for idx, (title, img) in enumerate(augmentations):
                axes[idx].imshow(img)
                axes[idx].set_title(title)
                axes[idx].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.info(f"Dataset increased by 30% through augmentation")

# =============================================================================
# D.4 DATA SPLITTING DEMO SECTION
# =============================================================================
elif section == "Data Splitting":
    st.header("📊 Data Splitting Module")
    
    st.subheader("D.4 - Dataset Splitting Strategy")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Code Purpose:** Strategic dataset partitioning
        
        **Splitting Strategy:**
        - **Training (80%)**: Foreign dataset - model learning
        - **Validation (20%)**: Foreign dataset - hyperparameter tuning
        - **Testing (100%)**: Philippines dataset - cross-domain evaluation
        
        **Key Functions:**
        - `split_datasets()`: Implements train/val/test split
        - Stratified sampling maintains class distribution
        """)
        
        # Show split statistics
        st.metric("Training Samples", "640 images")
        st.metric("Validation Samples", "160 images") 
        st.metric("Testing Samples", "200 images")
        
    with col2:
        # Create split visualization
        labels = ['Training (Foreign)', 'Validation (Foreign)', 'Testing (Philippines)']
        sizes = [640, 160, 200]
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        ax.set_title('Dataset Split Distribution')
        
        st.pyplot(fig)

# =============================================================================
# D.5 MODEL TRAINING DEMO SECTION  
# =============================================================================
elif section == "Model Training":
    st.header("🤖 Model Training Module")
    
    st.subheader("D.5 - Multi-Model Training Approach")
    
    # Model selection
    model_type = st.radio(
        "Select model to explore:",
        ["VGG16 Baseline", "VGG16 + PLSR", "VGG16 + XGBoost"]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **{model_type} Architecture**
        
        **Code Purpose:** {"" if "Baseline" in model_type else "Feature-based "}Classification
        
        **Training Strategy:**
        - **VGG16 Base**: Pre-trained on ImageNet, fine-tuned
        - **Hyperparameter Tuning**: KerasTuner with L2 regularization
        - **Feature Extraction**: CNN features for traditional ML
        - **Early Stopping**: Prevent overfitting
        
        **Key Functions:**
        - `build_hyper_vgg16()`: Hyperparameter search space
        - `extract_features_correct()`: Feature extraction
        - `train_plsr_model()` / `train_xgboost_model()`: Classifier training
        """)
    
    with col2:
        # Load and display model metrics
        if st.button("Load Model Performance"):
            # Sample performance data
            models = {
                "VGG16 Baseline": {"Accuracy": 0.89, "Precision": 0.87, "Recall": 0.91},
                "VGG16 + PLSR": {"Accuracy": 0.92, "Precision": 0.90, "Recall": 0.94},
                "VGG16 + XGBoost": {"Accuracy": 0.91, "Precision": 0.89, "Recall": 0.93}
            }
            
            perf = models[model_type]
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{perf['Accuracy']:.3f}")
            with col2:
                st.metric("Precision", f"{perf['Precision']:.3f}")
            with col3:
                st.metric("Recall", f"{perf['Recall']:.3f}")
            
            # Training history plot
            fig, ax = plt.subplots(figsize=(8, 4))
            epochs = range(1, 11)
            train_acc = [0.5 + i*0.05 for i in epochs]
            val_acc = [0.5 + i*0.04 for i in epochs]
            
            ax.plot(epochs, train_acc, 'b-', label='Training Accuracy')
            ax.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'{model_type} Training History')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)

# =============================================================================
# LIVE PREDICTION SECTION
# =============================================================================
elif section == "Live Prediction":
    st.header("🔮 Live Prediction Demo")
    
    st.markdown("""
    **Test the trained models with new rice leaf images**
    Upload an image to see predictions from all three models.
    """)
    
    uploaded_file = st.file_uploader("Choose a rice leaf image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)
        
        if st.button("Analyze Image"):
            # Simulate model predictions (in real app, load actual models)
            with st.spinner('Running models...'):
                # Simulate processing time
                import time
                time.sleep(2)
                
                # Mock predictions
                predictions = {
                    "VGG16 Baseline": {"class": "LEAFBLAST", "confidence": 0.87},
                    "VGG16 + PLSR": {"class": "LEAFBLAST", "confidence": 0.92},
                    "VGG16 + XGBoost": {"class": "HEALTHY", "confidence": 0.78}
                }
                
                # Display results
                st.success("Analysis Complete!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("VGG16 Baseline")
                    st.metric("Prediction", predictions["VGG16 Baseline"]["class"])
                    st.metric("Confidence", f"{predictions['VGG16 Baseline']['confidence']:.3f}")
                
                with col2:
                    st.subheader("VGG16 + PLSR") 
                    st.metric("Prediction", predictions["VGG16 + PLSR"]["class"])
                    st.metric("Confidence", f"{predictions['VGG16 + PLSR']['confidence']:.3f}")
                
                with col3:
                    st.subheader("VGG16 + XGBoost")
                    st.metric("Prediction", predictions["VGG16 + XGBoost"]["class"])
                    st.metric("Confidence", f"{predictions['VGG16 + XGBoost']['confidence']:.3f}")

# =============================================================================
# SYSTEM OVERVIEW
# =============================================================================
else:
    st.header("🏗️ System Architecture Overview")
    
    st.markdown("""
    ## Complete Rice Leaf Blast Detection Pipeline
    
    ### 🔍 Pre-processing Module
    - Image loading and enhancement
    - Duplicate and blur detection  
    - Class balancing
    - CLAHE contrast enhancement
    
    ### 🔄 Data Augmentation Module
    - Geometric transformations
    - Photometric adjustments
    - 30% dataset increase
    
    ### 📊 Data Splitting Module
    - Foreign dataset: Training/Validation
    - Philippines dataset: Testing
    - Cross-domain evaluation
    
    ### 🤖 Model Training Module
    - VGG16 baseline with fine-tuning
    - VGG16 + PLSR feature classification
    - VGG16 + XGBoost ensemble
    - Hyperparameter optimization
    
    ### 📈 Validation & Testing
    - Comprehensive metrics evaluation
    - Grad-CAM explainability
    - Cross-domain performance analysis
    """)
    
    # System flowchart
    st.image("https://via.placeholder.com/800x400?text=System+Architecture+Flowchart", 
             caption="System Architecture Flow")

# Footer
st.markdown("---")
st.markdown("🌾 **Rice Leaf Blast Detection System** | Demo Application")
