@echo off

cd /d "C:\Users\Admin\Rahul\islt_directml\Pre-Processing\Pickle maker"

:activate_environment
call activate slt_cuda
if errorlevel 1 (
    echo Failed to activate the environment. Exiting...
    exit /b
)


start python -m Checkpointed_Pickle_Create0
start python -m Checkpointed_Pickle_Create1
start python -m Checkpointed_Pickle_Create2
start python -m Checkpointed_Pickle_Create3
start python -m Checkpointed_Pickle_Create4
start python -m Checkpointed_Pickle_Create5
start python -m Checkpointed_Pickle_Create6
start python -m Checkpointed_Pickle_Create7
start python -m Checkpointed_Pickle_Create8
start python -m Checkpointed_Pickle_Create9


:end
