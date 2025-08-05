import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np
import plotly.express as px
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="🍎 FreshHarvest Classifier",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .prediction-container {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #4CAF50;
    }
    .fresh-fruit {
        background: linear-gradient(45deg, #4CAF50, #8BC34A);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .spoiled-fruit {
        background: linear-gradient(45deg, #f44336, #ff9800);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(45deg, #4CAF50, #8BC34A);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Class names and mapping
CLASSES = ['F_Banana', 'F_Lemon', 'F_Lulo', 'F_Mango', 'F_Orange', 'F_Strawberry',
           'F_Tamarillo', 'F_Tomato', 'S_Banana', 'S_Lemon', 'S_Lulo', 'S_Mango',
           'S_Orange', 'S_Strawberry', 'S_Tamarillo', 'S_Tomato']

FRUIT_EMOJIS = {
    'Banana': '🍌', 'Lemon': '🍋', 'Lulo': '🟠', 'Mango': '🥭',
    'Orange': '🍊', 'Strawberry': '🍓', 'Tamarillo': '🍅', 'Tomato': '🍅'
}


@st.cache_resource
def load_model():
    """Load the trained ResNet-50 model"""
    try:
        # Initialize model architecture
        model = resnet50(pretrained=False)

        # Recreate the custom classifier head
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, len(CLASSES))
        )

        # Load trained weights
        try:
            model.load_state_dict(torch.load('best_resnet50_cpu.pth', map_location='cpu'))
            st.success("✅ Model loaded successfully!")
        except FileNotFoundError:
            st.error(
                "❌ Model file 'best_resnet50_cpu.pth' not found. Please ensure the model file is in the same directory.")
            st.info("💡 You can train the model using the provided training script first.")
            return None

        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None


def preprocess_image(image):
    """Preprocess the uploaded image for model prediction"""
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Apply transformations
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor


def predict_image(model, image_tensor):
    """Make prediction on the preprocessed image"""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class_idx].item()

    return predicted_class_idx, confidence, probabilities[0]


def parse_prediction(class_name):
    """Parse the prediction to extract fruit type and freshness"""
    parts = class_name.split('_')
    freshness = "Fresh" if parts[0] == 'F' else "Spoiled"
    fruit_type = parts[1]
    emoji = FRUIT_EMOJIS.get(fruit_type, '🍎')

    return freshness, fruit_type, emoji


def create_confidence_chart(probabilities, classes):
    """Create a confidence chart for top predictions"""
    # Get top 5 predictions
    top_indices = torch.topk(probabilities, 5).indices
    top_probs = torch.topk(probabilities, 5).values

    data = []
    for idx, prob in zip(top_indices, top_probs):
        class_name = classes[idx.item()]
        freshness, fruit_type, emoji = parse_prediction(class_name)
        display_name = f"{emoji} {freshness} {fruit_type}"
        data.append({
            'Class': display_name,
            'Confidence': prob.item() * 100,
            'Type': freshness
        })

    df = pd.DataFrame(data)

    # Create horizontal bar chart
    fig = px.bar(
        df,
        x='Confidence',
        y='Class',
        color='Type',
        color_discrete_map={'Fresh': '#4CAF50', 'Spoiled': '#f44336'},
        title="Top 5 Predictions with Confidence Scores",
        labels={'Confidence': 'Confidence (%)', 'Class': 'Prediction'}
    )

    fig.update_layout(
        height=400,
        showlegend=True,
        yaxis={'categoryorder': 'total ascending'}
    )

    return fig


# Main App
def main():
    st.markdown('<h1 class="main-header">🍎 FreshHarvest Classifier</h1>', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; font-size: 1.2rem; margin-bottom: 2rem;">
        Upload an image of a fruit and let AI determine if it's <span style="color: #4CAF50; font-weight: bold;">Fresh</span> 
        or <span style="color: #f44336; font-weight: bold;">Spoiled</span>! 🔍
    </div>
    """, unsafe_allow_html=True)

    # Sidebar information
    with st.sidebar:
        st.header("📋 About the Model")
        st.write("""
        - **Architecture**: ResNet-50 with transfer learning
        - **Classes**: 16 fruit types (Fresh & Spoiled)
        - **Input Size**: 128×128 pixels
        - **Supported Fruits**: Banana, Lemon, Lulo, Mango, Orange, Strawberry, Tamarillo, Tomato
        """)

        with st.expander("🎯 Supported Classes"):
            fresh_fruits = [cls for cls in CLASSES if cls.startswith('F_')]
            spoiled_fruits = [cls for cls in CLASSES if cls.startswith('S_')]

            st.write("**Fresh Fruits:**")
            for fruit in fresh_fruits:
                fruit_name = fruit.split('_')[1]
                emoji = FRUIT_EMOJIS.get(fruit_name, '🍎')
                st.write(f"{emoji} {fruit_name}")

            st.write("**Spoiled Fruits:**")
            for fruit in spoiled_fruits:
                fruit_name = fruit.split('_')[1]
                emoji = FRUIT_EMOJIS.get(fruit_name, '🍎')
                st.write(f"{emoji} {fruit_name}")

    # Load model
    model = load_model()

    if model is None:
        st.stop()

    # File uploader with drag and drop
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        uploaded_file = st.file_uploader(
            "🖼️ Drag and drop an image or click to upload",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image of a fruit to classify its freshness"
        )

    if uploaded_file is not None:
        # Create two columns for image and results
        img_col, result_col = st.columns([1, 1])

        with img_col:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Add image info
            st.write(f"**Image Size**: {image.size}")
            st.write(f"**Format**: {image.format}")

        with result_col:
            # Make prediction
            with st.spinner("🔄 Analyzing image..."):
                try:
                    # Preprocess and predict
                    image_tensor = preprocess_image(image)
                    predicted_idx, confidence, all_probabilities = predict_image(model, image_tensor)

                    # Parse prediction
                    predicted_class = CLASSES[predicted_idx]
                    freshness, fruit_type, emoji = parse_prediction(predicted_class)

                    # Display main prediction
                    if freshness == "Fresh":
                        st.markdown(f"""
                        <div class="fresh-fruit">
                            {emoji} {freshness} {fruit_type}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="spoiled-fruit">
                            {emoji} {freshness} {fruit_type}
                        </div>
                        """, unsafe_allow_html=True)

                    # Display confidence
                    st.markdown(f"""
                    <div class="prediction-container">
                        <h3>🎯 Prediction Details</h3>
                        <p><strong>Fruit Type:</strong> {fruit_type} {emoji}</p>
                        <p><strong>Condition:</strong> {freshness}</p>
                        <p><strong>Confidence:</strong> {confidence * 100:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")
                    st.stop()

        # Display confidence chart
        st.plotly_chart(
            create_confidence_chart(all_probabilities, CLASSES),
            use_container_width=True
        )

        # Additional insights
        st.markdown("### 💡 Insights")

        insight_col1, insight_col2 = st.columns(2)

        with insight_col1:
            if confidence > 0.8:
                st.success("🎯 **High Confidence**: The model is very confident about this prediction!")
            elif confidence > 0.6:
                st.warning("⚠️ **Medium Confidence**: The prediction is likely correct but consider the image quality.")
            else:
                st.error("❓ **Low Confidence**: Please try a clearer image for better results.")

        with insight_col2:
            if freshness == "Fresh":
                st.info("✅ This fruit appears to be in good condition and safe to consume!")
            else:
                st.warning("⚠️ This fruit shows signs of spoilage. Consider discarding it.")

    else:
        # Show example when no image is uploaded
        st.markdown("### 📸 How to use:")
        st.markdown("""
        1. **Upload an image** using the drag-and-drop area above
        2. **Wait for analysis** - our AI model will process your image
        3. **View results** - see the freshness prediction with confidence scores
        4. **Interpret results** - use the insights to make informed decisions
        """)

        # Add sample images section
        st.markdown("### 🍎 Try with sample images:")
        st.info("💡 **Tip**: For best results, use clear, well-lit images with the fruit as the main subject.")


if __name__ == "__main__":
    main()