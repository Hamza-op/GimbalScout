param(
    [string]$Destination = "$PSScriptRoot\..\assets",
    [string]$FfmpegUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
    [string]$YoloUrl = $env:YOLO_ONNX_URL
)

$ErrorActionPreference = "Stop"

$Destination = [System.IO.Path]::GetFullPath($Destination)
$tmpRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("video-tool-assets-" + [System.Guid]::NewGuid().ToString("N"))

New-Item -ItemType Directory -Force -Path $Destination | Out-Null
New-Item -ItemType Directory -Force -Path $tmpRoot | Out-Null

try {
    $ffmpegArchive = Join-Path $Destination "ffmpeg-tools.zip"

    if (Test-Path $ffmpegArchive) {
        Write-Host "ffmpeg tools archive already exists in $Destination"
    } else {
        $archive = Join-Path $tmpRoot "ffmpeg.zip"
        $extractDir = Join-Path $tmpRoot "ffmpeg"
        $toolDir = Join-Path $tmpRoot "ffmpeg-tools"

        Write-Host "Downloading ffmpeg tools from $FfmpegUrl"
        Invoke-WebRequest -Uri $FfmpegUrl -OutFile $archive

        Write-Host "Extracting ffmpeg tools"
        Expand-Archive -Path $archive -DestinationPath $extractDir -Force

        $downloadedFfmpeg = Get-ChildItem -Path $extractDir -Recurse -Filter "ffmpeg.exe" | Select-Object -First 1
        $downloadedFfprobe = Get-ChildItem -Path $extractDir -Recurse -Filter "ffprobe.exe" | Select-Object -First 1

        if ($null -eq $downloadedFfmpeg -or $null -eq $downloadedFfprobe) {
            throw "Downloaded ffmpeg archive did not contain ffmpeg.exe and ffprobe.exe"
        }

        New-Item -ItemType Directory -Force -Path $toolDir | Out-Null
        Copy-Item -LiteralPath $downloadedFfmpeg.FullName -Destination (Join-Path $toolDir "ffmpeg.exe") -Force
        Copy-Item -LiteralPath $downloadedFfprobe.FullName -Destination (Join-Path $toolDir "ffprobe.exe") -Force

        Write-Host "Compressing ffmpeg tools archive"
        Compress-Archive -LiteralPath (Join-Path $toolDir "ffmpeg.exe"), (Join-Path $toolDir "ffprobe.exe") -DestinationPath $ffmpegArchive -CompressionLevel Optimal -Force
    }

    if ($YoloUrl) {
        $yolo = Join-Path $Destination "yolo.onnx"
        if (Test-Path $yolo) {
            Write-Host "YOLO model already exists in $Destination"
        } else {
            Write-Host "Downloading YOLO model from $YoloUrl"
            Invoke-WebRequest -Uri $YoloUrl -OutFile $yolo
        }
    } else {
        Write-Host "YOLO_ONNX_URL not set; skipping optional yolo.onnx download"
    }

    Get-ChildItem -Path $Destination | Select-Object Name, Length | Format-Table -AutoSize
} finally {
    Remove-Item -LiteralPath $tmpRoot -Recurse -Force -ErrorAction SilentlyContinue
}
