@echo off

cd /d "C:\Users\Admin\Rahul\islt_directml\Pre-Processing\Pickle maker"

:activate_environment
call activate slt_cuda
if errorlevel 1 (
    echo Failed to activate the environment. Exiting...
    exit /b
)


start CUDA_VISIBLE_DEVICES=0 python -m Checkpointed_Pickle_Create_TEST_Only
start CUDA_VISIBLE_DEVICES=1 python -m Checkpointed_Pickle_Create_VAL_Only
start CUDA_VISIBLE_DEVICES=0 python -m Checkpointed_Pickle_Create0
start CUDA_VISIBLE_DEVICES=1 python -m Checkpointed_Pickle_Create1
start CUDA_VISIBLE_DEVICES=0 python -m Checkpointed_Pickle_Create2
start CUDA_VISIBLE_DEVICES=1 python -m Checkpointed_Pickle_Create3
start CUDA_VISIBLE_DEVICES=0 python -m Checkpointed_Pickle_Create4
start CUDA_VISIBLE_DEVICES=1 python -m Checkpointed_Pickle_Create5
start CUDA_VISIBLE_DEVICES=0 python -m Checkpointed_Pickle_Create6
start CUDA_VISIBLE_DEVICES=1 python -m Checkpointed_Pickle_Create7
start CUDA_VISIBLE_DEVICES=0 python -m Checkpointed_Pickle_Create8
start CUDA_VISIBLE_DEVICES=1 python -m Checkpointed_Pickle_Create9


:end
