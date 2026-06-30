@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "ROOT=%%~fI"

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss"') do set "RUN_ID=%%I"

if not defined HGT_CONDA_ENV set "HGT_CONDA_ENV=HGT"
if not defined HGT_GRAPH_DIR set "HGT_GRAPH_DIR=%ROOT%\data\hgt_graph"
if not defined HGT_VIZ_OUTPUT_DIR set "HGT_VIZ_OUTPUT_DIR=%ROOT%\outputs\preprocess\hgt_graph_viz\runs\%RUN_ID%"

echo [visualize_hgt_graph] env=%HGT_CONDA_ENV%
echo [visualize_hgt_graph] graph-dir=%HGT_GRAPH_DIR%
echo [visualize_hgt_graph] output-dir=%HGT_VIZ_OUTPUT_DIR%

conda run -n "%HGT_CONDA_ENV%" python "%SCRIPT_DIR%visualize_hgt_graph.py" ^
  --graph-dir "%HGT_GRAPH_DIR%" ^
  --output-dir "%HGT_VIZ_OUTPUT_DIR%" ^
  %*

if errorlevel 1 (
  echo [visualize_hgt_graph] failed
  exit /b %errorlevel%
)

echo [visualize_hgt_graph] completed: %HGT_VIZ_OUTPUT_DIR%
endlocal
