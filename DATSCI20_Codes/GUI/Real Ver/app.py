# app.py
import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import Model
import time

# Set page configuration
st.set_page_config(
    page_title="Rice Leaf Blast Detection System",
    page_icon="🌾",
    layout="wide"
)

# Load models with caching
@st.cache_resource
def load_models():
    """Load all trained models from .pkl files"""
    models = {}
    
    try:
        # Load baseline VGG16 model
        if os.path.exists('models/baseline_vgg16_regularized.pkl'):
            models['baseline'] = joblib.load('models/baseline_vgg16_regularized.pkl')
            st.sidebar.success("✅ Baseline VGG16 loaded")
        else:
            st.sidebar.error("❌ Baseline model not found")
            
        # Load PLSR model and scaler
        if os.path.exists('models/plsr_model.pkl') and os.path.exists('models/plsr_scaler.pkl'):
            models['plsr_model'] = joblib.load('models/plsr_model.pkl')
            models['plsr_scaler'] = joblib.load('models/plsr_scaler.pkl')
            st.sidebar.success("✅ PLSR model loaded")
        else:
            st.sidebar.warning("⚠️ PLSR model or scaler not found")
            
        # Load XGBoost model and scaler
        if os.path.exists('models/xgboost_model.pkl') and os.path.exists('models/xgboost_scaler.pkl'):
            models['xgboost_model'] = joblib.load('models/xgboost_model.pkl')
            models['xgboost_scaler'] = joblib.load('models/xgboost_scaler.pkl')
            st.sidebar.success("✅ XGBoost model loaded")
        else:
            st.sidebar.warning("⚠️ XGBoost model or scaler not found")
            
        # Load validation results if available
        if os.path.exists('models/validation_results.pkl'):
            models['validation_results'] = joblib.load('models/validation_results.pkl')
            
        return models
        
    except Exception as e:
        st.sidebar.error(f"❌ Error loading models: {e}")
        return {}

# Feature extraction function (from your notebook)
def extract_features_correct(baseline_model, x_data):
    """
    Extracts features from the baseline VGG16 model - same as your notebook
    """
    try:
        # The 'baseline_model' is Sequential, and its first layer (index 0) is the VGG16 base
        feature_extractor_model = baseline_model.layers[0]
        
        # Predict to get the features
        features = feature_extractor_model.predict(x_data, batch_size=16, verbose=0)
        return features
    except Exception as e:
        st.error(f"Feature extraction error: {e}")
        return None

# Image preprocessing function (from your notebook)
def load_and_preprocess_image(image, target_size=(224, 224)):
    """Load and preprocess image for CNN - same as your notebook"""
    try:
        # Convert PIL Image to array
        image = image.resize(target_size)
        image_array = np.array(image)

        # Ensure 3 channels (same as your notebook)
        if len(image_array.shape) == 2:  # Grayscale
            image_array = np.stack([image_array] * 3, axis=-1)
        elif image_array.shape[2] == 4:  # RGBA
            image_array = image_array[:, :, :3]

        return image_array.astype(np.float32) / 255.0  # Normalize to [0,1]
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# CLAHE enhancement (from your notebook)
def apply_clahe_enhancement(image_array):
    """Apply CLAHE enhancement to RGB image - same as your notebook"""
    try:
        # Convert to 8-bit for OpenCV processing
        img_uint8 = (image_array * 255).astype(np.uint8)

        # Convert to LAB color space to apply CLAHE on luminance channel
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE on the luminance channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)

        # Merge back the enhanced luminance with original a and b channels
        lab_clahe = cv2.merge([l_clahe, a, b])

        # Convert back to RGB
        enhanced_rgb = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

        # Convert back to float and normalize
        result_image = enhanced_rgb.astype(np.float32) / 255.0

        return result_image

    except Exception as e:
        st.error(f"Error in CLAHE enhancement: {e}")
        return image_array

# Title and description
st.title("🌾 Rice Leaf Blast Detection System")
st.markdown("""
This interactive demo showcases the complete pipeline for detecting rice leaf blast disease 
using computer vision and machine learning with real trained models.
""")

# Load models at startup
if 'models_loaded' not in st.session_state:
    with st.spinner("Loading trained models..."):
        st.session_state.models = load_models()
        st.session_state.models_loaded = True

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Select Demo Section:",
    ["System Overview", "Pre-processing", "Data Augmentation", 
     "Data Splitting", "Model Training", "Live Prediction"]
)

# Display loaded models in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Loaded Models")
if st.session_state.models:
    for model_name in ['baseline', 'plsr_model', 'xgboost_model']:
        if model_name in st.session_state.models:
            st.sidebar.write(f"✅ {model_name}")
else:
    st.sidebar.write("❌ No models loaded")

# =============================================================================
# LIVE PREDICTION SECTION (USES ACTUAL MODELS)
# =============================================================================
if section == "Live Prediction":
    st.header("🔮 Live Prediction with Real Models")
    
    st.markdown("""
    **Test the actual trained models with new rice leaf images**
    Upload an image to see real predictions from all three trained models.
    """)
    
    uploaded_file = st.file_uploader("Choose a rice leaf image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
        with col2:
            # Preprocessing options
            st.subheader("Preprocessing Options")
            apply_clahe = st.checkbox("Apply CLAHE Enhancement", value=True)
            show_attention = st.checkbox("Show Model Attention", value=True)
        
        if st.button("🔍 Analyze Image with All Models", type="primary"):
            with st.spinner('Processing image through all trained models...'):
                # Start timing
                start_time = time.time()
                
                # Preprocess image
                processed_image = load_and_preprocess_image(image)
                
                if processed_image is not None:
                    # Apply CLAHE if selected
                    if apply_clahe:
                        processed_image = apply_clahe_enhancement(processed_image)
                    
                    # Add batch dimension
                    image_batch = np.expand_dims(processed_image, axis=0)
                    
                    # Get predictions from each model
                    predictions = {}
                    
                    # 1. Baseline VGG16 prediction
                    if 'baseline' in st.session_state.models:
                        try:
                            baseline_pred = st.session_state.models['baseline'].predict(image_batch, verbose=0)[0][0]
                            predictions['VGG16 Baseline'] = {
                                'class': 'LEAFBLAST' if baseline_pred > 0.5 else 'HEALTHY',
                                'confidence': float(baseline_pred) if baseline_pred > 0.5 else float(1 - baseline_pred),
                                'raw_score': float(baseline_pred),
                                'inference_time': time.time() - start_time
                            }
                        except Exception as e:
                            st.error(f"Baseline model error: {e}")
                    
                    # 2. PLSR prediction
                    if 'plsr_model' in st.session_state.models and 'plsr_scaler' in st.session_state.models:
                        try:
                            # Extract features
                            features = extract_features_correct(st.session_state.models['baseline'], image_batch)
                            
                            if features is not None:
                                # Flatten features
                                if len(features.shape) > 2:
                                    features = features.reshape(features.shape[0], -1)
                                
                                # Scale features
                                features_scaled = st.session_state.models['plsr_scaler'].transform(features)
                                
                                # Predict
                                plsr_pred = st.session_state.models['plsr_model'].predict(features_scaled)[0][0]
                                predictions['VGG16 + PLSR'] = {
                                    'class': 'LEAFBLAST' if plsr_pred > 0.5 else 'HEALTHY',
                                    'confidence': float(plsr_pred) if plsr_pred > 0.5 else float(1 - plsr_pred),
                                    'raw_score': float(plsr_pred),
                                    'inference_time': time.time() - start_time
                                }
                        except Exception as e:
                            st.error(f"PLSR model error: {e}")
                    
                    # 3. XGBoost prediction
                    if 'xgboost_model' in st.session_state.models and 'xgboost_scaler' in st.session_state.models:
                        try:
                            # Extract features
                            features = extract_features_correct(st.session_state.models['baseline'], image_batch)
                            
                            if features is not None:
                                # Flatten features
                                if len(features.shape) > 2:
                                    features = features.reshape(features.shape[0], -1)
                                
                                # Scale features
                                features_scaled = st.session_state.models['xgboost_scaler'].transform(features)
                                
                                # Predict
                                xgb_pred_proba = st.session_state.models['xgboost_model'].predict_proba(features_scaled)[0][1]
                                predictions['VGG16 + XGBoost'] = {
                                    'class': 'LEAFBLAST' if xgb_pred_proba > 0.5 else 'HEALTHY',
                                    'confidence': float(xgb_pred_proba) if xgb_pred_proba > 0.5 else float(1 - xgb_pred_proba),
                                    'raw_score': float(xgb_pred_proba),
                                    'inference_time': time.time() - start_time
                                }
                        except Exception as e:
                            st.error(f"XGBoost model error: {e}")
                    
                    # Display results
                    st.success(f"✅ Analysis Complete! Processed {len(predictions)} models")
                    
                    # Create results columns
                    st.subheader("📊 Model Predictions")
                    cols = st.columns(len(predictions))
                    
                    for idx, (model_name, pred) in enumerate(predictions.items()):
                        with cols[idx]:
                            # Determine color based on confidence
                            confidence_color = "#28a745" if pred['confidence'] > 0.8 else "#ffc107" if pred['confidence'] > 0.6 else "#dc3545"
                            
                            # Create a nice card display
                            st.markdown(f"""
                            <div style="border: 2px solid {confidence_color}; border-radius: 10px; padding: 20px; text-align: center; margin: 10px;">
                                <h3>{model_name}</h3>
                                <h2 style="color: {confidence_color};">{pred['class']}</h2>
                                <div style="background-color: #f8f9fa; border-radius: 5px; padding: 10px;">
                                    <h4 style="margin: 0;">{pred['confidence']:.1%}</h4>
                                    <small>Confidence</small>
                                </div>
                                <div style="margin-top: 10px;">
                                    <small>Raw score: {pred['raw_score']:.3f}</small><br>
                                    <small>Time: {pred['inference_time']:.2f}s</small>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Show consensus analysis
                    st.subheader("🤝 Model Consensus Analysis")
                    
                    blast_votes = sum(1 for p in predictions.values() if p['class'] == 'LEAFBLAST')
                    total_models = len(predictions)
                    consensus_ratio = blast_votes / total_models
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("LEAFBLAST Votes", f"{blast_votes}/{total_models}")
                    
                    with col2:
                        st.metric("Consensus Ratio", f"{consensus_ratio:.1%}")
                    
                    with col3:
                        if consensus_ratio == 1.0:
                            st.error("🚨 ALL MODELS DETECT: LEAF BLAST DISEASE")
                        elif consensus_ratio == 0.0:
                            st.success("✅ ALL MODELS AGREE: HEALTHY LEAF")
                        elif consensus_ratio >= 0.67:
                            st.warning("⚠️ STRONG CONSENSUS: LIKELY LEAF BLAST")
                        elif consensus_ratio >= 0.33:
                            st.info("🔍 MIXED OPINION: INCONCLUSIVE")
                        else:
                            st.success("✅ MAJORITY: HEALTHY LEAF")
                    
                    # Confidence distribution
                    st.subheader("📈 Confidence Distribution")
                    model_names = list(predictions.keys())
                    confidences = [p['confidence'] for p in predictions.values()]
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    bars = ax.bar(model_names, confidences, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
                    ax.set_ylabel('Confidence Score')
                    ax.set_ylim(0, 1)
                    ax.set_title('Model Confidence Scores')
                    
                    # Add value labels on bars
                    for bar, confidence in zip(bars, confidences):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{confidence:.3f}', ha='center', va='bottom')
                    
                    # Add threshold line
                    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Decision Threshold')
                    ax.legend()
                    
                    st.pyplot(fig)
                    
                    # Raw prediction scores
                    with st.expander("📋 Detailed Raw Scores"):
                        scores_data = {
                            'Model': model_names,
                            'Raw Score': [p['raw_score'] for p in predictions.values()],
                            'Decision Threshold': ['> 0.5 = LEAFBLAST'] * len(predictions)
                        }
                        scores_df = pd.DataFrame(scores_data)
                        st.dataframe(scores_df)
                        
                else:
                    st.error("❌ Failed to process image. Please try another image.")

# =============================================================================
# MODEL TRAINING SECTION WITH REAL METRICS
# =============================================================================
elif section == "Model Training":
    st.header("🤖 Model Training with Real Performance")
    
    # Try to load actual performance data from validation results
    actual_performance = None
    if 'validation_results' in st.session_state.models:
        actual_performance = st.session_state.models['validation_results']
    
    st.subheader("D.5 - Actual Model Performance")
    
    model_type = st.radio(
        "Select model to inspect:",
        ["VGG16 Baseline", "VGG16 + PLSR", "VGG16 + XGBoost"]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if model_type == "VGG16 Baseline":
            st.markdown("""
            **VGG16 Baseline - Actual Performance**
            
            **Architecture:** Fine-tuned VGG16 with custom classification head
            **Training:** Hyperparameter tuning with KerasTuner + L2 regularization
            """)
            
        elif model_type == "VGG16 + PLSR":
            st.markdown("""
            **VGG16 + PLSR - Actual Performance**
            
            **Architecture:** VGG16 feature extraction + PLS Regression
            **Advantage:** Better generalization on small datasets
            """)
            
        else:  # XGBoost
            st.markdown("""
            **VGG16 + XGBoost - Actual Performance**
            
            **Architecture:** VGG16 feature extraction + XGBoost classifier
            **Advantage:** Handles complex feature relationships
            """)
    
    with col2:
        if st.button("📊 Load Actual Performance Metrics"):
            if actual_performance:
                # Display actual metrics from your validation
                model_key = model_type.split()[-1].lower()
                
                if model_key in actual_performance:
                    metrics = actual_performance[model_key]
                    
                    st.subheader(f"✅ Real Validation Metrics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
                    with col2:
                        st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
                    with col3:
                        st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
                    with col4:
                        st.metric("F1-Score", f"{metrics.get('f1_score', 0):.3f}")
                    
                    # Show confusion matrix if available
                    if 'predictions' in metrics and 'confusion_matrix' not in metrics:
                        # Create simple confusion matrix visualization
                        st.info("Real validation metrics loaded from your trained models!")
                else:
                    st.warning("Performance data not available for this model")
            else:
                st.warning("""
                **Validation results not loaded**
                
                To see real metrics, make sure you have:
                - `validation_results.pkl` in your models folder
                - This file is generated by your notebook during validation
                """)

# =============================================================================
# OTHER SECTIONS (Keep your existing code for other sections)
# =============================================================================
elif section == "Pre-processing":
    st.header("🔍 Pre-processing Module")
    # ... [keep your existing pre-processing code] ...

elif section == "Data Augmentation":
    st.header("🔄 Data Augmentation Module") 
    # ... [keep your existing augmentation code] ...

elif section == "Data Splitting":
    st.header("📊 Data Splitting Module")
    # ... [keep your existing splitting code] ...

else:  # System Overview
    st.header("🏗️ System Architecture Overview")
    
    # Show model loading status
    if st.session_state.models:
        st.success("✅ All models loaded successfully!")
        
        # Display model info
        st.subheader("Loaded Model Information")
        model_info = []
        
        if 'baseline' in st.session_state.models:
            baseline_model = st.session_state.models['baseline']
            model_info.append({
                'Model': 'VGG16 Baseline',
                'Type': 'Sequential CNN',
                'Layers': len(baseline_model.layers),
                'Status': '✅ Loaded'
            })
        
        if 'plsr_model' in st.session_state.models:
            model_info.append({
                'Model': 'VGG16 + PLSR', 
                'Type': 'Feature-based Regression',
                'Layers': 'N/A',
                'Status': '✅ Loaded'
            })
            
        if 'xgboost_model' in st.session_state.models:
            model_info.append({
                'Model': 'VGG16 + XGBoost',
                'Type': 'Feature-based Ensemble', 
                'Layers': 'N/A',
                'Status': '✅ Loaded'
            })
            
        if model_info:
            st.table(pd.DataFrame(model_info))
    else:
        st.error("❌ Models failed to load. Check your .pkl files.")

# Footer
st.markdown("---")
st.markdown("🌾 **Rice Leaf Blast Detection System** | Real Model Integration v2.0")
