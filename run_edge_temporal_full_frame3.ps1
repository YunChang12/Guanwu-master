$ErrorActionPreference = "Stop"

$logDir = "E:\QingYan\Guanwu-master2\workspace\projects\video\demo_master2_video_main\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stdout = Join-Path $logDir "edge_temporal_full_frame3_stdout.log"
$stderr = Join-Path $logDir "edge_temporal_full_frame3_stderr.log"
$marker = Join-Path $logDir "edge_temporal_full_frame3_launcher_marker.txt"

try {
  "launcher_start=$(Get-Date -Format o); pid=$PID" | Set-Content -LiteralPath $marker -Encoding UTF8

  $env:PYTHONPATH = "E:\QingYan\Guanwu-master2\src"
  $env:PYTHONIOENCODING = "utf-8"
  $env:PYTHONDONTWRITEBYTECODE = "1"
  $env:GUANWU_POSE_OPTIMIZER_MODE = "edge_contour_fast_temporal"
  $env:GUANWU_POSE_TARGET_FRAME_ID = "3"
  $env:GUANWU_POSE_TARGET_WINDOW_RADIUS = "2"
  Remove-Item Env:\GUANWU_POSE_TARGET_OBJECT_IDS -ErrorAction SilentlyContinue
  Remove-Item Env:\GUANWU_POSE_MAX_TARGET_OBJECTS -ErrorAction SilentlyContinue

  Set-Location "E:\QingYan\Guanwu-master2"

  & "D:\AnacondaPackage\Anaconda\envs\3d_env\python.exe" "run_video_pipeline.py" `
    "--video" "E:\QingYan\Guanwu-master2\workspace\raw\video.mp4" `
    "--project-name" "demo_master2_video_main" `
    "--from-stage" "pose.optimize" `
    "--to-stage" "scene.compose" `
    "--force" `
    1> $stdout `
    2> $stderr

  "launcher_exit=$(Get-Date -Format o); exitcode=$LASTEXITCODE" | Add-Content -LiteralPath $marker -Encoding UTF8
  exit $LASTEXITCODE
}
catch {
  "launcher_error=$(Get-Date -Format o); $($_.Exception.Message)" | Add-Content -LiteralPath $marker -Encoding UTF8
  $_ | Out-String | Add-Content -LiteralPath $stderr -Encoding UTF8
  exit 1
}
