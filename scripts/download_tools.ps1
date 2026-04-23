param(
    [string]$Destination = "$PSScriptRoot\..\assets"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $Destination | Out-Null

Write-Host "Download helper scaffold"
Write-Host "Destination: $Destination"
Write-Host ""
Write-Host "Add your ffmpeg / ffprobe / yolo.onnx download commands here."
Write-Host "This script exists so the bootstrap path is explicit instead of implied."
