CXX = g++
CXXFLAGS = -fopenmp -Wall -ansi -std=c++2a
INC = -I$(CPYTHONPATH)
LIB_PATH= -L$(PYTHONPATH)
LIB = -lpython39

data: 
	$(CXX) $(INC) $(LIB_PATH) $(LIB) -O2 $(CXXFLAGS) Finite_Difference_Solver.h Finite_Difference_Solver.cpp Get_Training_Data.cpp -o get_data
	
train: 
	$(CXX) $(INC) $(LIB_PATH) $(LIB) -O2 $(CXXFLAGS) Finite_Difference_Solver.h Finite_Difference_Solver.cpp Train_Agent.cpp -o train_agent