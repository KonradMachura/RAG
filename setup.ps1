# Bookipidia Setup Script for Windows

Write-Host "--- Starting Bookipidia Setup ---" -ForegroundColor Cyan

# 1. Create virtual environment
if (-not (Test-Path .venv)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Gray
    python -m venv .venv
} else {
    Write-Host "Virtual environment already exists." -ForegroundColor Gray
}

# 2. Activate and Install dependencies
Write-Host "Installing/Updating dependencies (this may take a few minutes)..." -ForegroundColor Gray
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\pip.exe install -r requirements.txt

# 3. Create necessary data directories
Write-Host "Creating data directories..." -ForegroundColor Gray
$dirs = @(
    "data/vector_db",
    "data/sources/documents/raw",
    "data/sources/documents/processed"
)
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -Path $dir -ItemType Directory -Force | Out-Null
    }
}

# 4. Handle .env file
if (-not (Test-Path .env)) {
    Write-Host "Creating .env from template..." -ForegroundColor Yellow
    Copy-Item .env.example .env
    Write-Host "ACTION REQUIRED: Please open the .env file and add your GROQ_API_KEY." -ForegroundColor Cyan
} else {
    Write-Host ".env file already exists." -ForegroundColor Gray
}

Write-Host "`nSetup Complete!" -ForegroundColor Green
Write-Host "To start the application:" -ForegroundColor White
Write-Host "1. Activate venv: .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "2. Run Backend: uvicorn src.api.main:app --reload" -ForegroundColor White
Write-Host "3. Run Frontend: streamlit run src/frontend/app.py" -ForegroundColor White
