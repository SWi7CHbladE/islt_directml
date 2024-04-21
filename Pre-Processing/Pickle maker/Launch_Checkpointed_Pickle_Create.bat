@echo off

cd /d "C:\Users\Admin\Rahul\islt_directml\Pre-Processing\Pickle maker"

:activate_environment
call activate slt_cuda
if errorlevel 1 (
    echo Failed to activate the environment. Exiting...
    exit /b
)

:check_files
if exist "C:\Users\Admin\Rahul\islt_directml\Pre-Processing\Pickle maker\Dataset\Pickles\excel_data.dev" if exist "C:\Users\Admin\Rahul\islt_directml\Pre-Processing\Pickle maker\Dataset\Pickles\excel_data.test" if exist "C:\Users\Admin\Rahul\islt_directml\Pre-Processing\Pickle maker\Dataset\Pickles\excel_data.train" (
    echo All files exist. Exiting...
    exit /b
)

:loop
python -m Checkpointed_Pickle_Create
WMIC process where name="python.exe" CALL setpriority "high"
if %errorlevel% equ 0 (
    echo Process completed successfully.
    goto :end
) else (
    echo Process crashed. Relaunching...
    timeout /t 1 >nul 2>nul  # Adjust this delay as needed
    goto :loop
)

:end
