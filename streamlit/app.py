# app.py
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0
from pathlib import Path
import plotly.graph_objects as go

st.set_page_config(
    page_title="Cervical Cell Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

#custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .result-box {
        background-color: #d4edda;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: 600;
        padding: 0.75rem;
        border-radius: 8px;
    }
    .cell-description {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

CLASSES = [
    "Dyskeratotic",
    "Koilocytotic",
    "Metaplastic",
    "Parabasal",
    "Superficial-Intermediate",
]

CELL_INFO = {
    "Dyskeratotic": {
        "description": "Abnormal keratinization of cells, often indicating pre-cancerous changes",
        "significance": "May indicate cervical intraepithelial neoplasia (CIN) or dysplasia",
        "characteristics": "Dense cytoplasm, irregular nucleus, premature keratinization",
        "risk": "High"
    },
    "Koilocytotic": {
        "description": "Cells showing features of HPV infection with perinuclear clearing",
        "significance": "Indicates Human Papillomavirus (HPV) infection",
        "characteristics": "Perinuclear halo, enlarged nucleus, irregular nuclear membrane",
        "risk": "Moderate to High"
    },
    "Metaplastic": {
        "description": "Cells undergoing transformation in the transformation zone",
        "significance": "Normal cellular adaptation, but requires monitoring",
        "characteristics": "Immature squamous cells, may show reactive changes",
        "risk": "Low"
    },
    "Parabasal": {
        "description": "Immature cells from the basal layer of the epithelium",
        "significance": "May indicate atrophy, inflammation, or hormonal changes",
        "characteristics": "Small cells with high nuclear-to-cytoplasmic ratio",
        "risk": "Low to Moderate"
    },
    "Superficial-Intermediate": {
        "description": "Mature squamous cells from the upper epithelial layers",
        "significance": "Generally normal cells, represent healthy cervical epithelium",
        "characteristics": "Large, flat cells with small nuclei, abundant cytoplasm",
        "risk": "Normal"
    }
}

DEVICE = torch.device("cpu")
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "efficientnet_b0_cervical_3way_split.pth"

@st.cache_resource
def load_model():
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 5)
    
    state_dict = torch.load(MODEL_PATH,map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

def predict(image: Image.Image):
    tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        conf, idx = torch.max(probs, dim=1)
    
    all_probs = probs[0].numpy()
    return CLASSES[idx.item()], conf.item(), all_probs

def create_probability_chart(probs):
    fig = go.Figure(data=[
        go.Bar(
            x=probs * 100,
            y=CLASSES,
            orientation='h',
            marker=dict(
                color=['#28a745' if p==max(probs) else '#1f77b4' for p in probs],
                line=dict(color='white', width=2)
            ),
            text=[f'{p:.1f}%' for p in probs * 100],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Classification Probabilities",
        xaxis_title="Confidence (%)",
        yaxis_title="Cell Type",
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

# Sidebar
with st.sidebar:
    st.markdown("### 🔬 About This Tool")
    st.markdown("""
    This AI-powered screening tool classifies cervical cells from Pap smear images 
    to assist healthcare professionals in detecting abnormalities.
    
    **Model:** EfficientNet-B0  
    **Dataset:** SIPaKMeD  
    **Classes:** 5 cell types
    """)
    
    st.markdown("---")
    st.markdown("### ℹ️ How to Use")
    st.markdown("""
    1. Upload a cropped Pap smear cell image
    2. Click 'Analyze Cell' to run classification
    3. Review the prediction and confidence score
    4. Read the detailed interpretation
    """)
    
    st.markdown("---")
    st.markdown("### 📚 Learn More")
    if st.button("View Cell Type Guide"):
        st.session_state.show_guide = True

# Main content
st.markdown('<p class="main-header">🔬 Cervical Cell Classification System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Assisted Pap Smear Screening | EfficientNet-B0 Architecture</p>', unsafe_allow_html=True)

st.markdown("""
<div class="warning-box">
    ⚠️ <strong>Medical Disclaimer:</strong> This tool is designed for research and screening assistance only. 
    It is NOT a substitute for professional medical diagnosis. All findings should be reviewed by qualified 
    cytopathologists or healthcare providers.
</div>
""", unsafe_allow_html=True)

# Create tabs for better organization
tab1, tab2, tab3 = st.tabs(["📤 Upload & Analyze", "📖 Cell Type Reference", "ℹ️ About Cervical Cancer"])

with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Upload Cell Image")
        uploaded_file = st.file_uploader(
            "Select a Pap smear cell image",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Upload a cropped image containing a single cervical cell"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("🔍 Analyze Cell", type="primary"):
                with st.spinner("Analyzing cell morphology..."):
                    label, confidence, all_probs = predict(image)
                    st.session_state.prediction = {
                        'label': label,
                        'confidence': confidence,
                        'probs': all_probs
                    }
    
    with col2:
        st.markdown("### Analysis Results")
        
        if 'prediction' in st.session_state:
            pred = st.session_state.prediction
            
            st.markdown(f"""
            <div class="result-box">
                <h3 style="margin-top: 0;">Classification: {pred['label']}</h3>
                <p style="font-size: 1.2rem; margin: 0;">
                    <strong>Confidence:</strong> {pred['confidence']:.1%}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence indicator
            if pred['confidence'] > 0.8:
                st.success("✅ High confidence prediction")
            elif pred['confidence'] > 0.6:
                st.warning("⚠️ Moderate confidence - consider additional review")
            else:
                st.error("❌ Low confidence - manual review strongly recommended")
            
            # Probability chart
            fig = create_probability_chart(pred['probs'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Cell information
            st.markdown("### 📋 Cell Type Information")
            info = CELL_INFO[pred['label']]
            
            st.markdown(f"""
            <div class="info-box">
                <p><strong>Description:</strong> {info['description']}</p>
                <p><strong>Clinical Significance:</strong> {info['significance']}</p>
                <p><strong>Key Characteristics:</strong> {info['characteristics']}</p>
                <p><strong>Risk Level:</strong> <span style="color: {'red' if info['risk'] in ['High', 'Moderate to High'] else 'orange' if 'Moderate' in info['risk'] else 'green'};">{info['risk']}</span></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Recommendations
            st.markdown("### 💡 Recommendations")
            if pred['label'] in ["Dyskeratotic", "Koilocytotic"]:
                st.warning("""
                **Action Required:**
                - Further cytological evaluation recommended
                - Consider HPV testing
                - Follow-up colposcopy may be indicated
                - Consult with gynecologist for management plan
                """)
            elif pred['label'] in ["Parabasal"]:
                st.info("""
                **Suggested Follow-up:**
                - Review clinical context (age, hormonal status)
                - Monitor for inflammation or infection
                - Standard screening interval may apply
                """)
            else:
                st.success("""
                **General Guidance:**
                - Continue routine screening schedule
                - Maintain regular gynecological check-ups
                """)
        else:
            st.info("👆 Upload an image and click 'Analyze Cell' to see results")

with tab2:
    st.markdown("## 📖 Cervical Cell Type Reference Guide")
    
    for cell_type, info in CELL_INFO.items():
        with st.expander(f"**{cell_type}** - {info['risk']} Risk"):
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.markdown(f"**Description:** {info['description']}")
                st.markdown(f"**Clinical Significance:** {info['significance']}")
                st.markdown(f"**Characteristics:** {info['characteristics']}")
            with col_b:
                risk_color = 'red' if info['risk'] in ['High', 'Moderate to High'] else 'orange' if 'Moderate' in info['risk'] else 'green'
                st.markdown(f"### Risk Level")
                st.markdown(f"<h2 style='color: {risk_color};'>{info['risk']}</h2>", unsafe_allow_html=True)

with tab3:
    st.markdown("## ℹ️ Understanding Cervical Cancer Screening")
    
    col_x, col_y = st.columns(2)
    
    with col_x:
        st.markdown("""
        ### What is a Pap Smear?
        
        A Pap smear (Pap test) is a screening procedure for cervical cancer. It tests for 
        the presence of precancerous or cancerous cells on the cervix.
        
        ### Why It Matters
        
        - **Early Detection:** Identifies abnormal cells before they become cancerous
        - **Prevention:** Allows treatment of precancerous conditions
        - **Life-Saving:** Regular screening reduces cervical cancer mortality by 60-90%
        
        ### Risk Factors
        
        - HPV infection (most significant)
        - Smoking
        - Weakened immune system
        - Multiple sexual partners
        - Early sexual activity
        """)
    
    with col_y:
        st.markdown("""
        ### Screening Guidelines
        
        **Ages 21-29:** Pap test every 3 years
        
        **Ages 30-65:** 
        - Pap test every 3 years, OR
        - HPV test every 5 years, OR
        - Combined Pap/HPV test every 5 years
        
        **Over 65:** May discontinue if previous tests were normal
        
        ### About This AI System
        
        This classifier uses deep learning (EfficientNet-B0) trained on the SIPaKMeD dataset 
        to identify five types of cervical cells. It serves as a decision support tool to 
        assist cytotechnologists and pathologists in screening workflow.
        
        **Limitations:**
        - Requires high-quality, properly prepared samples
        - Cannot replace expert human evaluation
        - Should be part of comprehensive diagnostic approach
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>Developed for research and educational purposes | Always consult healthcare professionals for medical decisions</p>
</div>
""", unsafe_allow_html=True)