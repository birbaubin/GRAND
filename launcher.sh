python_exe = "python3.9"
$python_exe -m pip install -r requirements.txt

$python_exe benchmark.py --dataset flickr --type P 
$python_exe benchmark.py --dataset flickr --type DP
$python_exe benchmark.py --dataset flickr --type DPD

