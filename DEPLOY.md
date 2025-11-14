# 🚀 Render Deployment Guide

## Quick Deploy to Render

1. **Go to [Render.com](https://render.com)** and sign up/login

2. **Create New Web Service:**
   - Click "New +" → "Web Service"
   - Connect your GitHub account
   - Select repository: `Fast-Delivery-Agent-Review-Classification`

3. **Configure Service:**
   - **Name:** `fast-delivery-ml` (or your choice)
   - **Environment:** `Python 3`
   - **Build Command:** `bash build.sh`
   - **Start Command:** `gunicorn api_simple:app`
   - **Instance Type:** Free (or paid for better performance)

4. **Click "Create Web Service"**

5. **Wait for Build** (5-10 minutes):
   - Installs dependencies
   - Generates dataset (2000 samples)
   - Trains TF-IDF model
   - Deploys your app

6. **Access Your App:**
   - Render will provide a URL like: `https://fast-delivery-ml.onrender.com`
   - Test at: `https://your-app.onrender.com`

## ⚙️ Environment Variables (Optional)

If needed, add in Render dashboard:
```
PYTHON_VERSION=3.13.0
```

## 📝 Notes

- **Free tier:** App sleeps after 15 min of inactivity (cold starts)
- **Paid tier ($7/mo):** Always on, faster performance
- **Models included:** TF-IDF trained automatically during build
- **DistilBERT:** Skipped in build (too slow), train locally if needed

## 🧪 Test Your Deployed API

```bash
curl -X POST https://your-app.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "Excellent service! Very professional driver."}'
```

## 🔍 Troubleshooting

**Build fails?**
- Check build logs in Render dashboard
- Verify `build.sh` has execute permissions
- Ensure all dependencies are in `requirements.txt`

**App not responding?**
- Free tier sleeps after inactivity (first request takes 30-60s)
- Check application logs in Render dashboard

## 📊 Monitor Your App

- **Logs:** Render Dashboard → Your Service → Logs
- **Metrics:** Dashboard shows CPU/Memory usage
- **Health:** Check `/health` endpoint

---

## Alternative: Railway Deployment

If Render doesn't work, try [Railway.app](https://railway.app):

1. Click "New Project" → "Deploy from GitHub"
2. Select your repository
3. Railway auto-detects Python and runs
4. No configuration needed!

---

Need help? Check Render docs: https://render.com/docs
