# Script de Execução Completa: Calibração + Detecção HDFS
# Salve como: run_hdfs_pipeline.ps1

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  HDFS Pipeline: Calibração + Detecção" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Navegar para diretório
Set-Location "D:\ProLog\03_HDFS_Benchmark"

# Verificar pré-requisitos
Write-Host "🔍 Verificando pré-requisitos..." -ForegroundColor Yellow

if (-not (Test-Path "saved_models\hdfs_loggpt.pt")) {
    Write-Host "❌ Modelo não encontrado: saved_models\hdfs_loggpt.pt" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path "saved_models\config.pt")) {
    Write-Host "❌ Config não encontrado: saved_models\config.pt" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Modelo encontrado (115MB)" -ForegroundColor Green
Write-Host ""

# ============================================
# ETAPA 1: CALIBRAÇÃO
# ============================================

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  ETAPA 1/2: CALIBRAÇÃO (100% sessões)" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "⏱️  Tempo estimado: ~2.5-3 horas" -ForegroundColor Yellow
Write-Host "📊 Sessões: 55.822 normais + 16.838 anômalas" -ForegroundColor Yellow
Write-Host ""

$startTime1 = Get-Date

# Verificar se já existe threshold_config.json
if (Test-Path "threshold_config.json") {
    Write-Host "⚠️  threshold_config.json já existe!" -ForegroundColor Yellow
    $response = Read-Host "Deseja re-calibrar? (s/n)"
    if ($response -ne "s") {
        Write-Host "⏭️  Pulando calibração..." -ForegroundColor Yellow
        goto DetectionStep
    }
}

Write-Host "🚀 Iniciando calibração..." -ForegroundColor Green
python calibrate_optimized.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Calibração falhou!" -ForegroundColor Red
    exit 1
}

$endTime1 = Get-Date
$duration1 = $endTime1 - $startTime1

Write-Host ""
Write-Host "✅ Calibração completa!" -ForegroundColor Green
Write-Host "⏱️  Tempo: $($duration1.ToString('hh\:mm\:ss'))" -ForegroundColor Green
Write-Host ""

# Exibir threshold encontrado
if (Test-Path "threshold_config.json") {
    $config = Get-Content "threshold_config.json" | ConvertFrom-Json
    Write-Host "📊 Threshold Calibrado:" -ForegroundColor Cyan
    Write-Host "   Threshold: $($config.threshold)" -ForegroundColor White
    Write-Host "   F1 Score: $($config.f1_score)" -ForegroundColor White
    Write-Host "   Precision: $($config.precision)" -ForegroundColor White
    Write-Host "   Recall: $($config.recall)" -ForegroundColor White
} else {
    Write-Host "❌ threshold_config.json não foi gerado!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Pressione ENTER para continuar para detecção..." -ForegroundColor Yellow
Read-Host

# ============================================
# ETAPA 2: DETECÇÃO
# ============================================

:DetectionStep

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  ETAPA 2/2: DETECÇÃO (100% sessões)" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "⏱️  Tempo estimado: ~1-1.5 horas" -ForegroundColor Yellow
Write-Host ""

$startTime2 = Get-Date

Write-Host "🚀 Iniciando detecção..." -ForegroundColor Green
python detect_chunked.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Detecção falhou!" -ForegroundColor Red
    exit 1
}

$endTime2 = Get-Date
$duration2 = $endTime2 - $startTime2

Write-Host ""
Write-Host "✅ Detecção completa!" -ForegroundColor Green
Write-Host "⏱️  Tempo: $($duration2.ToString('hh\:mm\:ss'))" -ForegroundColor Green
Write-Host ""

# ============================================
# RESULTADOS FINAIS
# ============================================

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  RESULTADOS FINAIS" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

if (Test-Path "results_chunked.txt") {
    Get-Content "results_chunked.txt"
} else {
    Write-Host "❌ results_chunked.txt não foi gerado!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  PIPELINE COMPLETO!" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

$totalDuration = $duration1 + $duration2
Write-Host "⏱️  Tempo Total: $($totalDuration.ToString('hh\:mm\:ss'))" -ForegroundColor Green
Write-Host ""
Write-Host "📁 Arquivos gerados:" -ForegroundColor Yellow
Write-Host "   - threshold_config.json" -ForegroundColor White
Write-Host "   - results_chunked.txt" -ForegroundColor White
Write-Host "   - detection_results_partial.pkl" -ForegroundColor White
Write-Host ""
Write-Host "✅ Envie esses arquivos para análise!" -ForegroundColor Green
