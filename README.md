<h1 align="center">🩺 AI-Powered Healthcare Intelligence Network</h1>

<p align="center">
  <strong>Revolutionizing Healthcare with AI-Driven Predictions, Recommendations, and Insights</strong>
  <br>
  <img src="https://source.unsplash.com/800x400/?healthcare,ai" alt="Healthcare AI" width="600">
</p>

---

<h2>📌 About This Project</h2>
<p>
  The <strong>AI-Powered Healthcare Intelligence Network</strong> is a cutting-edge platform that leverages Machine Learning (ML) and Natural Language Processing (NLP) to provide 
  accurate disease predictions, personalized medical recommendations, and AI-assisted drug suggestions. The system aims to enhance early diagnosis, reduce medical errors, and 
  offer intelligent healthcare solutions.
</p>

---

<h2>🚀 Features</h2>

<h3>💡 Disease Prediction & Medical Recommendation</h3>
<p>
  This module uses **Machine Learning** to predict diseases based on symptoms and suggest the best medical recommendations.
</p>
<ul>
  <li>✔️ Predicts diseases from given symptoms.</li>
  <li>✔️ Uses **RandomForest Classifier** for predictions.</li>
  <li>✔️ Provides recommended treatments and precautions.</li>
</ul>
<table>
  <tr>
    <td><img src="utils/disease_prediction.png" alt="Disease Prediction" width="300"></td>
  </tr>
</table>

<h3>💊 AI-Powered Drug Recommendation</h3>
<p>
  Our AI system uses **NLP & Cosine Similarity** to recommend alternative medicines based on drug properties.
</p>
<ul>
  <li>✔️ Matches medicines with similar ingredients.</li>
  <li>✔️ Ensures safer and more effective drug prescriptions.</li>
</ul>
<table>
  <tr>
    <td><img src="utils/drug_recommendation.png" alt="Drug Recommendation" width="300"></td>
  </tr>
</table>

<h3>🫀 Heart Disease Risk Assessment</h3>
<p>
  This module uses **LightGBM & AI classifiers** to assess heart disease risks based on patient history.
</p>
<ul>
  <li>✔️ Evaluates heart disease risk based on lifestyle and medical history.</li>
  <li>✔️ Provides AI-driven health improvement recommendations.</li>
</ul>
<table>
  <tr>
    <td><img src="utils/heart_risk.png" alt="Heart Disease Risk" width="300"></td>
  </tr>
</table>

<h3>🤖 Medibot - AI Health Assistant</h3>
<p>
  Our **LLM-powered chatbot** answers medical queries and provides instant healthcare insights using **Hugging Face LLM (Mistral-7B-Instruct)**.
</p>
<ul>
  <li>✔️ Retrieves medical information from a FAISS vector database.</li>
  <li>✔️ Provides **reliable AI-driven** answers to health-related questions.</li>
</ul>
<table>
  <tr>
    <td><img src="utils/medibot.png" alt="Medibot AI Assistant" width="300"></td>
  </tr>
</table>

---

<h2>📂 Folder Structure</h2>
<pre>
📦 AI-Powered Healthcare Intelligence Network
│── 📂 models/                         # Trained ML models
│── 📂 data/                           # Medical datasets (CSV)
│── 📂 vectorstore/db_faiss/           # FAISS vector database
│── 📂 utils/                          # Images, styles, and helper files
│── 📂 pages/                          # Individual module pages
│── 📜 home.py                         # Main homepage (Streamlit UI)
│── 📜 requirements.txt                 # Dependencies
│── 📜 README.md                        # Project Documentation
│── 📜 .gitignore                        # Ignored files
│── 📜 styles.css                        # Custom CSS for UI
</pre>

---

<h2>⚙️ Installation & Setup</h2>

<h3>1️⃣ Clone the Repository</h3>
<pre>
git clone https://github.com/AbhaySingh71/AI-Powered-Healthcare-Intelligence-System.git
cd AI-Powered-Healthcare-Intelligence-System
</pre>

<h3>2️⃣ Set Up the Virtual Environment</h3>
<pre>
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows
</pre>

<h3>3️⃣ Install Dependencies</h3>
<pre>
pip install -r requirements.txt
</pre>

<h3>4️⃣ Set Up Environment Variables</h3>
<p>Create a <code>.env</code> file and add:</p>
<pre>
HF_TOKEN=your_huggingface_api_token
</pre>
<p>Ensure it is added to GitHub Secrets when deploying.</p>

<h3>5️⃣ Run the Application</h3>
<pre>
streamlit run home.py
</pre>

---

<h2>🚀 Deployment on Streamlit Cloud</h2>
<h3>1️⃣ Push code to GitHub</h3>
<pre>
git add .
git commit -m "Initial commit"
git push origin main
</pre>

<h3>2️⃣ Deploy on Streamlit</h3>
<ul>
  <li>Go to <a href="https://share.streamlit.io/">Streamlit Cloud</a> → Deploy a new app.</li>
  <li>Set <code>HF_TOKEN</code> in Streamlit Secrets.</li>
  <li>Click <strong>Deploy!</strong> 🎉</li>
</ul>

---



<h2>⚙️ Technologies Used</h2>
<ul>
  <li><strong>Machine Learning:</strong> RandomForest, LightGBM, NLP, Cosine Similarity</li>
  <li><strong>AI & NLP:</strong> Hugging Face Transformers, LangChain, FAISS</li>
  <li><strong>Data Handling:</strong> Pandas, NumPy, Pickle</li>
  <li><strong>Web Framework:</strong> Streamlit</li>
  <li><strong>Visualization:</strong> Plotly, SHAP for feature importance</li>
  <li><strong>Cloud Deployment:</strong> AWS, GCP</li>
</ul>

---

<h2>🔍 Why Use This App?</h2>
<ul>
  <li>🏥 <strong>AI-Powered Healthcare Insights:</strong> Get data-driven medical predictions.</li>
  <li>⚕️ <strong>Enhances Patient Care:</strong> Supports doctors and patients in making informed decisions.</li>
  <li>💡 <strong>Real-Time Recommendations:</strong> Provides immediate AI-assisted insights.</li>
  <li>⏳ <strong>Saves Time:</strong> Automates diagnosis and medical recommendations.</li>
  <li>🔬 <strong>Empowers Medical Research:</strong> Helps in early disease detection and prevention.</li>
</ul>

---

<h2>📜 License</h2>
<p>
  This project is licensed under the <strong>MIT License</strong>. Feel free to use, modify, and contribute!
</p>

---

<h2>📬 Contact Us</h2>
<p>Have questions or need support? Reach out to us at:</p>
<ul>
  <li>📧 <a href="mailto:abhaysingh71711@gmail.com">abhaysingh71711@gmail.com</a></li>
</ul>

---

<h2>🌐 Connect With Me</h2>
<p align="center">
  <a href="https://github.com/abhaysingh71711" target="_blank">🐙 GitHub</a> |
  <a href="https://linkedin.com/in/abhaysingh71711" target="_blank">🔗 LinkedIn</a> |
  <a href="https://twitter.com/abhaysingh71711" target="_blank">🐦 Twitter</a>
</p>
