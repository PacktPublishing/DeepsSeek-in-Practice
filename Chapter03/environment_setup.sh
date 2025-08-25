#!/bin/bash
# Environment setup script for Chapter 3

echo "Setting up Chapter 3 development environment..."

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    cat > .env << EOF
# DeepSeek Configuration
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

# Database Configuration
DATABASE_URL=sqlite:///./development.db

# API Configuration
API_HOST=localhost
API_PORT=8000

# Logging
LOG_LEVEL=INFO

# Security
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
EOF
    echo "Created .env file. Please update with your actual API keys."
else
    echo ".env file already exists."
fi

# Create .gitignore
cat > .gitignore << EOF
# Environment variables
.env

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Database
*.db
*.sqlite3

# Logs
*.log

# OS
.DS_Store
Thumbs.db
EOF

echo "Environment setup complete!"
echo "Next steps:"
echo "1. Update .env file with your DeepSeek API key"
echo "2. Install dependencies: pip install -r 01-code-assistant/requirements.txt"
echo "3. Open the project in Cursor IDE"
