#!/bin/bash
# GitHub Deployment Script for FAISS Educational App

echo \"🚀 Deploying FAISS Educational App to GitHub...\"

# Add the deployment guide to git
git add DEPLOYMENT.md
git commit -m \"Add deployment guide and setup files\"

# Check if repository exists and push
echo \"📤 Attempting to push to GitHub...\"
if git push -u origin main; then
    echo \"✅ Successfully pushed to GitHub!\"
    echo \"\"
    echo \"🌐 Next steps:\"
    echo \"1. Visit: https://github.com/signupwithanand/vectorDB-FAISSProj\"
    echo \"2. Go to: https://share.streamlit.io/\"
    echo \"3. Deploy your app with the repository URL\"
    echo \"\"
    echo \"📖 Read DEPLOYMENT.md for detailed instructions\"
else
    echo \"❌ Push failed. Repository might not exist yet.\"
    echo \"\"
    echo \"📋 Manual steps:\"
    echo \"1. Create repository at: https://github.com/new\"
    echo \"2. Name it: vectorDB-FAISSProj\"
    echo \"3. Make it public\"
    echo \"4. Run: git push -u origin main\"
    echo \"\"
    echo \"📖 See DEPLOYMENT.md for detailed instructions\"
fi

echo \"\"
echo \"🎉 Your FAISS Educational App is ready for deployment!\"
echo \"📁 Repository: https://github.com/signupwithanand/vectorDB-FAISSProj\"
echo \"🌐 Will be live at: https://share.streamlit.io/signupwithanand/vectordb-faissproj/main/app.py\"