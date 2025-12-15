# Push Fire/Smoke Training to New GitHub Repository

## Step 1: Configure Git (One-time setup)

```bash
# Set your git identity
git config --global user.name "whamidouche"
git config --global user.email "whamidouche@microsoft.com"

# Configure SSH to use your GitHub key
cat > ~/.ssh/config << 'SSHEOF'
Host github.com
  HostName github.com
  User git
  IdentityFile /home/whamidouche/ssdprivate/.ssh/github_key
  IdentitiesOnly yes
SSHEOF

chmod 600 ~/.ssh/config

# Test SSH connection
ssh -T git@github.com
# You should see: "Hi whamidouche! You've successfully authenticated..."
```

## Step 2: Create New Repository on GitHub

Go to: https://github.com/new

- Repository name: `fire-smoke-detection-yolov8`
- Description: `Minimal YOLOv8 training framework for fire/smoke detection`
- **Keep it Private** (unless you want public)
- **DO NOT** initialize with README (we already have one)

## Step 3: Initialize and Push Your Project

```bash
cd /home/whamidouche/ssdprivate/fire_detection/CEVG-RTNet/fire_smoke_training

# Initialize git repository
git init
git add .
git commit -m "Initial commit: Clean YOLOv8 fire/smoke detection framework"

# Link to your new GitHub repo (replace YOUR_USERNAME)
git remote add origin git@github.com:whamidouche/fire-smoke-detection-yolov8.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Step 4: Verify

Visit: https://github.com/whamidouche/fire-smoke-detection-yolov8

You should see:
- train.py
- test.py
- analyze.py
- dataset.yaml
- README.md

Done! 🎉

## Alternative: Using HTTPS instead of SSH

If SSH doesn't work, use HTTPS with a personal access token:

```bash
# Create token at: https://github.com/settings/tokens
# Select: repo (full control)

git remote remove origin
git remote add origin https://github.com/whamidouche/fire-smoke-detection-yolov8.git
git push -u origin main
# Enter your GitHub username and the token as password
```
