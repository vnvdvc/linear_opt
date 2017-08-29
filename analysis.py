import re,sys
from linear_opt import Model,Initialize
import matplotlib.pyplot as plt
class Model(object):
  def __init__(self,text):
  #text is a list of three lines
    self.text = ''.join(text)
    self.loss = float(re.findall('\d+\.\d+e?[+\-]?\d+',text[len(text)-1])[0])

  def __cmp__(self,other):
    if self.loss == other.loss:
      return 0
    else:
      return (self.loss - other.loss)/abs(self.loss-other.loss)

  def __repr__(self):
    return self.text

def graph(x0,params,data):
  cutoff = 10
  initial = Initialize(x0,cutoff)
  taud = 2.6
  model = Model(taud,params,initial,et_model=fet)
  times = data[:,0]
  fig, ax = plt.subplots()
  real = ax.scatter(times,data[:,1])
  simu = ax.plot(times,model.qt(times))
  plt.show()



if __name__ == "__main__":
  '''
  filename = sys.argv[1]
  linenum = 3
  models = []
  with open(filename,"r") as inp:
    line = inp.readline()
    while(line):
      if line.find("nitial condition x0s") >= 0:
        tempText = []
        for idx in range(linenum):
          tempText.append(line)
          line = inp.readline()
        models.append(Model(tempText))
      else:
        line = inp.readline()
  models = sorted(models)
  numOfPrints = 10
  for model in models[:numOfPrints]:
    print model
  '''
  x0 = 0.86
  mutant_files = ["y98r.txt","y98h.txt","y98a.txt","y98f.txt"]
  params = np.array([1.33,1.06,0.84])
  with open("./coding/fld/"+mutant_files[0],"r") as inp:
    data = []
    for line in inp:
      nums = line[:-1].split()
      data.append((float(nums[0]),float(nums[1])))

  graph(x0,params,np.array(data))








