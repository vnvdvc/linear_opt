import csv, subprocess

num_of_mutants = 4
initials = reversed(range(2**num_of_mutants))
for index in initials:
  qsub_command = """qsub -v INDEX={} myscript""".format(index)
  exit_status = subprocess.call(qsub_command,shell=True)
  if exit_status is 1:
    print "Job {} failed to submit".format(qsub_command)
print "Done submitting jobs!"
