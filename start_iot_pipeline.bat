@echo off
REM Start script for iot_pipeline using conda env 'tf-env'
SET CONDA_ROOT=C:\ProgramData\Anaconda3
IF EXIST "%CONDA_ROOT%\condabin\conda.bat" (
    call "%CONDA_ROOT%\condabin\conda.bat" activate tf-env
) ELSE (
    echo Could not find conda at %CONDA_ROOT%. Ensure CONDA_ROOT is correct or activate your env manually.
)

REM By default run in dry-run safe mode. Remove --dry-run --no-mqtt to run listener.
python iot_pipeline.py --dry-run --run-pipeline --no-mqtt

REM Deactivate environment (optional)
call conda deactivate

echo Done.
pause
