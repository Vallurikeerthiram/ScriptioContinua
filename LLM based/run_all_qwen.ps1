$projectRoot = "C:\Users\kamma\OneDrive - Amrita vishwa vidyapeetham\Project_Phase\LLM based"

Write-Host "Running Type 1..."
python "$projectRoot\type 1\run_type1.py" --model qwen3.5:4b --resume

Write-Host "Running Type 2..."
python "$projectRoot\type 2\run_type2.py" --model qwen3.5:4b --resume

Write-Host "Running Type 3..."
python "$projectRoot\type 3\run_type3.py" --model qwen3.5:4b --resume

Write-Host "All tasks completed!"
