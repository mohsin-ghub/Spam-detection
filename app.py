import streamlit as st
import pickle
import os

# Load the trained model and vectorizer safely
if os.path.exists("spam_model.pkl") and os.path.exists("tfidf_vectorizer.pkl"):
    model = pickle.load(open("spam_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
else:
    st.error("Error: Model or vectorizer file not found! Please upload them.")
    st.stop()  # Stop execution if files are missing

# Streamlit UI
st.title("📩 Spam Detection App")
st.write("🔍 Enter a message below to check if it's **Spam** or **Not Spam**.")

# Input text box
message = st.text_area("✍️ Enter your message:", height=100)

# Predict button
if st.button("🔎 Predict"):
    if message.strip() == "":
        st.warning("⚠️ Please enter a message to analyze.")
    else:
        # Transform inpust message
        message_vectorized = vectorizer.transform([message])
        
        # Predict
        prediction = model.predict(message_vectorized)[0]
        
        # Display result with better styling
        if prediction == 1:
            st.error("🚨 **SPAM DETECTED!** This message looks suspicious.")
        else:
            st.success("✅ **Not Spam!** This message seems safe.")

# Footer
st.markdown("---")
st.markdown("🛠 Built with **Streamlit & Machine Learning** | 🚀 *AI-Powered Spam Detection*")

