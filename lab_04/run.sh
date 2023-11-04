#!/bin/bash -l

#SBATCH --account=plgar2023-cpu
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=plgrid
#SBATCH --cpus-per-task=36
#SBATCH --mem-per-cpu=512M

module purge
module load scipy-bundle/2021.10-intel-2021b

# I do not think we want the tasks to share resources
# export SLURM_OVERLAP=1

problem_sizes=(256 512 1024)
side_length=256
cpu_min_count=1
cpu_max_count=36
series_count=5
theta=128
iters=64
csv_header="process_count,problem_size,series_id,time"
output_dir="${SCRATCH}/ar"
output_file_base="${output_dir}/$(date +%Y%m%dT%H%M%S)"
output_file="${output_dir}/$(date +%Y%m%dT%H%M%S).csv"


mkdir -p $output_dir
echo $csv_header > $output_file

total_task=$(( "${#problem_sizes[@]}" * (cpu_max_count - cpu_min_count + 1) * series_count ))
completed_task=0

for (( n_cpu = $cpu_min_count ; n_cpu <= $cpu_max_count ; n_cpu++ )); do
  for problem_size in "${problem_sizes[@]}"; do
    for (( series_id = 0 ; series_id < $series_count ; series_id++ )); do
      echo "Run: mpiexec -np $n_cpu ./main.py --series $series_id --side $side_length --theta $theta --iters 64 --points-per-proc $problem_size >> $output_file"
      mpiexec -np $n_cpu ./main.py --series $series_id --side $side_length --theta $theta --iters 64 --points-per-proc $problem_size >> $output_file
      completed_task=$(( $completed_task + 1 ))
      echo "Completion: $completed_task / $total_task"
    done
  done
done

zip -q "${output_file_base}.zip" $output_file
# rm -f $output_file

# mpiexec ./main.py 10 13 255 30

