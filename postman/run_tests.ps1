# Simple PowerShell script to run Postman tests
param(
    [string]$Environment = "local",
    [switch]$Detailed,
    [switch]$Help
)

if ($Help) {
    Write-Host "Usage: .\run_tests.ps1 [-Environment local|docker|production] [-Detailed] [-Help]"
    exit 0
}

Write-Host "Multimodal RAG API Test Runner" -ForegroundColor Cyan

# Check Newman
if (!(Get-Command newman -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Newman not installed. Run: npm install -g newman newman-reporter-htmlextra" -ForegroundColor Red
    exit 1
}

# Create results directory
$resultsDir = "..\test-results"
if (!(Test-Path $resultsDir)) {
    New-Item -ItemType Directory -Path $resultsDir | Out-Null
}

# Set base URL
$baseUrl = "http://localhost:8000"
if ($Environment -eq "docker") { $baseUrl = "http://localhost:8000" }
if ($Environment -eq "production") { $baseUrl = "https://your-production-url.com" }

Write-Host "Testing against: $baseUrl" -ForegroundColor Green

# Build command
$args = @(
    "run", "Multimodal_RAG_API.postman_collection.json",
    "--environment", "Multimodal_RAG_API.postman_environment.json",
    "--env-var", "base_url=$baseUrl",
    "--reporters", "cli,json",
    "--reporter-json-export", "$resultsDir\newman-results.json",
    "--delay-request", "1000",
    "--timeout", "30000"
)

if ($Detailed) {
    $args += @("--reporters", "cli,json,htmlextra", "--reporter-htmlextra-export", "$resultsDir\newman-report.html")
}

# Run tests
Write-Host "Running Newman tests..." -ForegroundColor Yellow
newman @args
exit $LASTEXITCODE
