#!/bin/bash -l
#SBATCH --account=plgar2023-cpu
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=plgrid
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=512M

module purge
module load scipy-bundle/2021.10-intel-2021b
module load vtune
# module add .plgrid plgrid/tools/openmpi

# I do not think we want the tasks to share resources
# export SLURM_OVERLAP=1

problem_sizes=(1024 2048 4096)
side_length=8192
cpu_min_count=32
cpu_max_count=32
series_count=1
theta=256
iters=64
csv_header="process_count,problem_size,series_id,time"
output_dir="${SCRATCH}/ar"
output_file_base="${output_dir}/$(date +%Y%m%dT%H%M%S)"
output_file="${output_dir}/$(date +%Y%m%dT%H%M%S).csv"

mkdir -p $output_dir
echo $csv_header > $output_file

total_task=$(( "${#problem_sizes[@]}" * (cpu_max_count - cpu_min_count + 1) * series_count ))
completed_task=0

# vtune hpc-performance
for (( n_cpu = $cpu_min_count ; n_cpu <= $cpu_max_count ; n_cpu++ )); do
  for problem_size in "${problem_sizes[@]}"; do
    for (( series_id = 0 ; series_id < $series_count ; series_id++ )); do
      vtune_output_dir="${output_dir}/vtune-hpc/cpu_${n_cpu}_size_${problem_size}_sid_${series_id}"
      mkdir -p $vtune_output_dir
      # echo "[$(date +%Y%m%dT%H%M%S)] Run: mpiexec -np $n_cpu ./main.py --series $series_id --side $side_length --theta $theta --iters 64 --grid-points $problem_size >> $output_file"
      mpiexec -np $n_cpu vtune -collect hpc-performance -trace-mpi -result-dir $vtune_output_dir -- ./main.py --series $series_id --side $side_length --theta $theta --iters 64 --grid-points $problem_size >> $output_file
      completed_task=$(( $completed_task + 1 ))
      echo "[$(date +%Y%m%dT%H%M%S)] Completion: $completed_task / $total_task"
    done
  done
done

# vtune hotspots
for (( n_cpu = $cpu_min_count ; n_cpu <= $cpu_max_count ; n_cpu++ )); do
  for problem_size in "${problem_sizes[@]}"; do
    for (( series_id = 0 ; series_id < $series_count ; series_id++ )); do
      vtune_output_dir="${output_dir}/vtune-hotspots/cpu_${n_cpu}_size_${problem_size}_sid_${series_id}"
      mkdir -p $vtune_output_dir
      # echo "[$(date +%Y%m%dT%H%M%S)] Run: mpiexec -np $n_cpu ./main.py --series $series_id --side $side_length --theta $theta --iters 64 --grid-points $problem_size >> $output_file"
      mpiexec -np $n_cpu vtune -collect hotspots -trace-mpi -result-dir $vtune_output_dir -- ./main.py --series $series_id --side $side_length --theta $theta --iters 64 --grid-points $problem_size >> $output_file
      completed_task=$(( $completed_task + 1 ))
      echo "[$(date +%Y%m%dT%H%M%S)] Completion: $completed_task / $total_task"
    done
  done
done

# for (( n_cpu = $cpu_min_count ; n_cpu <= $cpu_max_count ; n_cpu++ )); do
#   for problem_size in "${problem_sizes[@]}"; do
#     for (( series_id = 0 ; series_id < $series_count ; series_id++ )); do
#       aps_output_dir="${output_dir}/aps/cpu_${n_cpu}_size_${problem_size}_sid_${series_id}"
#       mkdir -p $aps_output_dir
#       # echo "[$(date +%Y%m%dT%H%M%S)] Run: mpiexec -np $n_cpu ./main.py --series $series_id --side $side_length --theta $theta --iters 64 --grid-points $problem_size >> $output_file"
#       mpiexec -np $n_cpu aps --result-dir=$aps_output_dir -c=mpi ./main.py --series $series_id --side $side_length --theta $theta --iters 64 --grid-points $problem_size >> $output_file
#       completed_task=$(( $completed_task + 1 ))
#       echo "[$(date +%Y%m%dT%H%M%S)] Completion: $completed_task / $total_task"
#     done
#   done
# done

# zip -q "${output_file_base}.zip" $output_file
# rm -f $output_file

# mpiexec ./main.py 10 13 255 30

