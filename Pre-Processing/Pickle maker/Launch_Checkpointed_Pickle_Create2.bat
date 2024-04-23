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
python -m Checkpointed_Pickle_Create0 && python -m Checkpointed_Pickle_Create1 && python -m Checkpointed_Pickle_Create2 && python -m Checkpointed_Pickle_Create3 && python -m Checkpointed_Pickle_Create4 && python -m Checkpointed_Pickle_Create5 && python -m Checkpointed_Pickle_Create6 && python -m Checkpointed_Pickle_Create7 && python -m Checkpointed_Pickle_Create8 && python -m Checkpointed_Pickle_Create9
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
