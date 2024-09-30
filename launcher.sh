python_exe=python3.9
# $python_exe -m pip install -r requirements.txt

$python_exe benchmark.py --dataset flickr --type P --n_experiments 1
$python_exe benchmark.py --dataset flickr --type DP --n_experiments 1
$python_exe benchmark.py --dataset flickr --type DPD --n_experiments 1

