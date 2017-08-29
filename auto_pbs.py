import csv, subprocess

num_of_mutants = 4
for index in [0,1]:
  for mutant in range(num_of_mutants):
    qsub_command = """qsub -v INDEX={},MUTANT={} myscript""".format(index,mutant)
    exit_status = subprocess.call(qsub_command,shell=True)
    if exit_status is 1:
      print "Job {} failed to submit".format(qsub_command)
print "Done submitting jobs!"
