import pickle

def print_status(namefile):
  print(" ")
  file_sys  = open(namefile+"_systems.npy", "rb")
  L = pickle.load(file_sys)
  file_sys.close()
  print("File: "+namefile)
  print(" ")
  print(" sys file  : " +str(len(L.keys())))
  file_prob = open(namefile+"_probability.npy", "rb")
  L = pickle.load(file_prob)
  file_prob.close()
  print(" porb file : " +str(len(L.keys())))
  
print_status("edkcbr_sano")
print_status("edkcbr_cancer")


